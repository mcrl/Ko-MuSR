import random
import ast
import io
from functools import partial

def process_docs(dataset):
    def _parse_choices(doc):
        out_doc = {
            "narrative": doc["narrative"],
            "question": doc["question"],
        }
        choice_str = doc["choices"]
        if isinstance(choice_str, str):
            choice_list = ast.literal_eval(choice_str)
        else:
            choice_list = choice_str
        out_doc["choices"] = choice_list
        if "reasoning" in doc:
            out_doc["reasoning"] = doc["reasoning"]
        if "answer_index" in doc:
            out_doc["answer_index"] = doc["answer_index"]
        if "answer_choice" in doc:
            out_doc["answer_choice"] = doc["answer_choice"]
        return out_doc
    return dataset.map(_parse_choices)

# doc to text section
## Common section
ko_default_direction="다음 선택지 중 하나를 고르세요:"
def ko_common(doc, middle_hint=None, direction=ko_default_direction):
    buf = io.StringIO()
    buf.write(doc["narrative"] + "\n\n")
    buf.write(doc["question"] + "\n\n")
    if middle_hint:
        buf.write(middle_hint + "\n\n")
    buf.write(direction + "\n")
    choices = ast.literal_eval(doc["choices"]) if isinstance(doc["choices"], str) else doc["choices"]
    for i, choice in enumerate(choices):
        buf.write(f"{i+1} - {choice}\n")
    buf.write("\n")
    return buf

en_default_direction="Choose one of the following options:"
def en_common(doc, middle_hint=None, direction=en_default_direction):
    buf = io.StringIO()
    buf.write(doc["narrative"] + "\n\n")
    buf.write(doc["question"] + "\n\n")
    if middle_hint:
        buf.write(middle_hint + "\n\n")
    buf.write(direction + "\n")
    choices = ast.literal_eval(doc["choices"]) if isinstance(doc["choices"], str) else doc["choices"]
    for i, choice in enumerate(choices):
        buf.write(f"{i+1} - {choice}\n")
    buf.write("\n")
    return buf

def mm_ta_common_routine(doc, cot=False, hint=False, **kwargs):
    isko = kwargs["det_func"](doc)
    if isko:
        hint_str = kwargs.get("ko_hint", None) if hint else None
        if cot:
            first = kwargs["ko_cot_direction_first"]
            last = kwargs["ko_cot_direction_last"]
        else:
            first = kwargs["ko_direct_direction_first"]
            last = kwargs["ko_direct_direction_last"]
        buf = ko_common(doc)
    else:
        hint_str = kwargs.get("en_hint", None) if hint else None
        if cot:
            first = kwargs["en_cot_direction_first"]
            last = kwargs["en_cot_direction_last"]
        else:
            first = kwargs["en_direct_direction_first"]
            last = kwargs["en_direct_direction_last"]
        buf = en_common(doc)
    if hint_str:
        buf.write(f"{first}\n\n{hint_str}\n\n{last}")
    else:
        buf.write(f"{first}\n{last}")
    return buf.getvalue()
        
    

def determinant(doc, keyword):
    return keyword in doc["question"]
is_komm = partial(determinant, keyword="살인자일 가능성이")
is_koop = partial(determinant, keyword="확인하겠는가")
is_kota = partial(determinant, keyword="배정하시겠습니까")


## MM Section
### MM Common

ko_mm_hint="""
살인자는 반드시 세 가지 조건을 충족해야 합니다:
동기(피해자를 죽일 만한 이유가 있는가), 수단(범행 도구에 접근할 수 있는가), 기회(범행 현장에 접근할 수 있었는가)
무고한 용의자는 이 세 가지 중 두 가지까지만 입증될 수 있으며, 의심스러운 행동을 보일 수는 있어도 세 가지 조건을 모두 갖추지는 않습니다. 만약 두 용의자 모두 수단, 동기, 기회를 가진 것처럼 보인다면, 그 중에서도 가장 명확하게 이 세 가지가 드러난 인물을 선택해야 합니다. 반대로, 어느 누구도 세 가지를 모두 갖추지 않은 경우라면, 세 가지 조건 중 가장 뚜렷하게 드러난 인물을 선택해야 합니다.
"""
ko_mm_cot_direction_first="""반드시 하나의 선택지만 고르셔야 합니다. 선택하기 전에, 단계별로 당신의 추론 과정을 설명하세요."""
ko_mm_direct_direction_first="""반드시 하나의 선택지만 고르셔야 합니다."""
ko_mm_cot_direction_last="""단계별로 논리적인 추론을 마친 후, 마지막에 아래와 같은 형식으로 정답을 작성하세요: '정답: X'"""
ko_mm_direct_direction_last="""마지막에 아래와 같은 형식으로 정답을 작성하세요: '정답: X'"""
en_mm_hint="""The murderer needs to have a means (access to weapon), motive (reason to kill the victim), and opportunity (access to crime scene) in order to have killed the victim. Innocent suspects may have two of these proven, but not all three. An innocent suspect may be suspicious for some other reason, but they will not have all of motive, means, and opportunity established.

If you believe that both suspects have motive, means, and opportunity, you should make an educated guess pick the one for whom these are best established. If you believe that neither suspect has all three established, then choose the suspect where these are most clearly established."""
en_mm_cot_direction_first="""You must pick one option. Before selecting a choice, explain your reasoning step by step."""
en_mm_direct_direction_first="""You must pick one option."""
en_mm_cot_direction_last="""Explain your reasoning step by step before you answer. Finally, the last thing you generate should be "ANSWER: (your answer here, including the choice number)"
"""
en_mm_direct_direction_last="""Finally, the last thing you generate should be "ANSWER: (your answer here, including the choice number)"
"""
mm_kwargs = {
    "det_func": is_komm,
    "ko_cot_direction_first": ko_mm_cot_direction_first,
    "ko_cot_direction_last": ko_mm_cot_direction_last,
    "ko_direct_direction_first": ko_mm_direct_direction_first,
    "ko_direct_direction_last": ko_mm_direct_direction_last,
    "ko_hint": ko_mm_hint,
    "en_cot_direction_first": en_mm_cot_direction_first,
    "en_cot_direction_last": en_mm_cot_direction_last,
    "en_direct_direction_first": en_mm_direct_direction_first,
    "en_direct_direction_last": en_mm_direct_direction_last,
    "en_hint": en_mm_hint,
}

def doc_to_text_mm_cot_hint(doc):
    return mm_ta_common_routine(doc, True, True, **mm_kwargs)
    
def doc_to_text_mm_cot_nohint(doc):
    return mm_ta_common_routine(doc, True, False, **mm_kwargs)

def doc_to_text_mm_direct_hint(doc):
    return mm_ta_common_routine(doc, False, True, **mm_kwargs)

def doc_to_text_mm_direct_nohint(doc):
    return mm_ta_common_routine(doc, False, False, **mm_kwargs)
    
## OP Section
ko_op_hint="""이야기를 바탕으로, 우리는 특정 인물이 어떤 물건이 어디에 있다고 생각하고 있는지를 파악하려고 합니다.
이를 위해서는 이야기 읽으며 각 인물이 그 물건의 위치를 시점별로 어떻게 인식하고 있는지를 추적해야 합니다.
물건이 옮겨질 때, 인물이 그 장면을 직접 보았거나, 옮겨진 위치를 인지할 수 있는 상태였다면 그들은 물건이 옮겨진 위치를 알고 있다고 봅니다.
하지만 물건이 움직일 때 인물이 그 장면을 보지 못했거나 다른 일에 집중하고 있었다면, 여전히 마지막으로 본 위치에 있다고 믿게 됩니다."""
en_op_hint="""Based on this story, we want to identify where someone believes that a certain object is at the end of the story. In order to do that, you need to read the story and keep track of where they think the object is at each point. When an object is moved, the person may observe its new location if they saw it move.

To see where an object ends up, they must be able to see the location that it moves to and not be too distracted by what they are doing. If they do not observe the object moving, then they will still believe it to be in the last location where they observed it."""

ko_op_cot_direction_first="""반드시 하나의 보기를 선택해야 합니다."""
ko_op_direct_direction_first=ko_op_cot_direction_first
ko_op_cot_direction_last="""추론 과정을 단계별로 설명한 후, 마지막에 다음 형식으로 정답을 출력하세요: '정답: X'"""
ko_op_direct_direction_last="""마지막에 다음 형식으로 정답을 출력하세요: '정답: X'"""
en_op_cot_direction_first="""You must pick one option."""
en_op_direct_direction_first=en_op_cot_direction_first
en_op_cot_direction_last="Explain your reasoning step by step, then output the answer in the form 'ANSWER: X'"
en_op_direct_direction_last="""Finally, the last thing you generate should be "ANSWER: (your answer here, include the choice number)"
"""
op_kwargs = {
    "det_func": is_koop,
    "ko_cot_direction_first": ko_op_cot_direction_first,
    "ko_cot_direction_last": ko_op_cot_direction_last,
    "ko_direct_direction_first": ko_op_direct_direction_first,
    "ko_direct_direction_last": ko_op_direct_direction_last,
    "ko_hint": ko_op_hint,
    "en_cot_direction_first": en_op_cot_direction_first,
    "en_cot_direction_last": en_op_cot_direction_last,
    "en_direct_direction_first": en_op_direct_direction_first,
    "en_direct_direction_last": en_op_direct_direction_last,
    "en_hint": en_op_hint,
}

def op_common(doc, cot=False, hint=False, reorder=False, **kwargs):
    isko = is_koop(doc)
    if isko:
        hint_str = kwargs.get("ko_hint", None) if hint else None
        if cot:
            first = kwargs["ko_cot_direction_first"]
            last = kwargs["ko_cot_direction_last"]
        else:
            first = kwargs["ko_direct_direction_first"]
            last = kwargs["ko_direct_direction_last"]
        buf = ko_common(doc) if reorder else ko_common(doc, middle_hint=hint_str)
    else:
        hint_str = kwargs.get("en_hint", None) if hint else None
        if cot:
            first = kwargs["en_cot_direction_first"]
            last = kwargs["en_cot_direction_last"]
        else:
            first = kwargs["en_direct_direction_first"]
            last = kwargs["en_direct_direction_last"]
        buf = en_common(doc) if reorder else en_common(doc, middle_hint=hint_str)
    if reorder and hint_str:
        buf.write(f"{first}\n\n{hint_str}\n\n{last}")
    else:
        buf.write(f"{first} {last}")
    return buf.getvalue()
        
        

def doc_to_text_op_cot_hint(doc):
    return op_common(doc, True, True, False, **op_kwargs)

def doc_to_text_op_cot_nohint(doc):
    return op_common(doc, True, False, False, **op_kwargs)
   
def doc_to_text_op_direct_hint(doc):
    return op_common(doc, False, True, False, **op_kwargs)

def doc_to_text_op_direct_nohint(doc):
    return op_common(doc, False, False, False, **op_kwargs)

def doc_to_text_op_cot_hint_reorder(doc):
    return op_common(doc, True, True, True, **op_kwargs)
    
def doc_to_text_op_direct_hint_reorder(doc):
    return op_common(doc, False, True, True, **op_kwargs)

## TA Section
ko_ta_hint = """스토리를 통해 각 사람이 한 가지 업무에 얼마나 능숙한지를 파악할 수 있습니다. 일반적으로 각 사람은 어떤 작업에 대해 뛰어나거나, 보통이거나, 부족한 실력을 가지고 있습니다. 우리는 가능한 한 각자의 강점을 최대한 발휘할 수 있도록 사람을 적절한 업무에 배치하려고 합니다.

또한, 두 사람이 함께 맡아야 하는 업무가 하나 있으며, 이 경우 두 사람의 팀워크 수준(훌륭함, 보통, 나쁨)도 전체 업무 성과에 중요한 영향을 미칩니다.

단, 두 사람이 함께 일해야 하는 작업에서 한 사람이 실력이 부족하고, 두 사람의 팀워크도 좋지 않다면, 다른 한 사람이 아무리 뛰어나도 전체 결과에는 도움이 되지 않을 수 있습니다.

각기 다른 강점, 약점, 그리고 두 사람 간의 상호작용을 고려하여, 전체 과제가 가장 효율적으로 수행될 수 있도록 팀원들을 적절히 배치해야 합니다."""
en_ta_hint = """The story should allow you to determine how good each person is at a skill. Roughly, each person is either great, acceptable, or bad at a task. We want to find an optimal assignment of people to tasks that uses their skills as well as possible. In addition, one task will have to have two people assigned to it. The effectiveness of their teamwork (great team, acceptable team, or bad team) also impacts the overall quality of the assignment.

When two people need to work on a task and one is bad at it, they don’t necessarily benefit from the other person being good, unless they work well together.

With different strengths, weaknesses, and interpersonal dynamics at play, you should allocate your team to find the single assignment to ensure that the tasks overall are completed as effectively as possible."""

ko_ta_cot_direction_first = "당신은 세 가지 중 하나의 선택지를 골라야 합니다."
ko_ta_direct_direction_first = ko_ta_cot_direction_first
ko_ta_cot_direction_last = """당신의 풀이 과정을 단계별로 설명한 뒤, 마지막에는 아래 형식으로 답변을 마무리해 주세요:  "정답: (당신의 선택, 번호 포함)
"""
ko_ta_direct_direction_last = """마지막에는 아래 형식으로 답변을 마무리해 주세요:  "정답: (당신의 선택, 번호 포함)"
"""

en_ta_cot_direction_first = "You must pick one option."
en_ta_direct_direction_first = en_ta_cot_direction_first
en_ta_cot_direction_last = """Explain your reasoning step by step before you answer. Finally, the last thing you generate should be "ANSWER: (your answer here, including the choice number)"
"""
en_ta_direct_direction_last = """Finally, the last thing you generate should be "ANSWER: (your answer here, including the choice number)"
"""
ta_kwargs = {
    "det_func": is_kota,
    "ko_cot_direction_first": ko_ta_cot_direction_first,
    "ko_cot_direction_last": ko_ta_cot_direction_last,
    "ko_direct_direction_first": ko_ta_direct_direction_first,
    "ko_direct_direction_last": ko_ta_direct_direction_last,
    "ko_hint": ko_ta_hint,
    "en_cot_direction_first": en_ta_cot_direction_first,
    "en_cot_direction_last": en_ta_cot_direction_last,
    "en_direct_direction_first": en_ta_direct_direction_first,
    "en_direct_direction_last": en_ta_direct_direction_last,
    "en_hint": en_ta_hint,
}

def doc_to_text_ta_cot_hint(doc):
    return mm_ta_common_routine(doc, True, True, **ta_kwargs)

def doc_to_text_ta_cot_nohint(doc):
    return mm_ta_common_routine(doc, True, False, **ta_kwargs)

def doc_to_text_ta_direct_hint(doc):
    return mm_ta_common_routine(doc, False, True, **ta_kwargs)

def doc_to_text_ta_direct_nohint(doc):
    return mm_ta_common_routine(doc, False, False, **ta_kwargs)

def has_answer(line):
    lower = line.lower()
    return "answer" in lower or "정답" in lower
def process_results(doc, results):
    """Take a single document and the LM results and evaluates, returning a
    dict where keys are the names of submetrics and values are the values of
    the metric for that one document

    :param doc:
        The document as returned from training_docs, validation_docs, or test_docs.
    :param results:
        The results of the requests created in construct_requests.
    """

    continuation = results[0]

    lines = continuation.split("\n")
    if len(lines) == 0:
        return {"acc": 0}

    answer_lines = [l for l in lines if has_answer(l)]

    if not answer_lines:
        return {"acc": 0}

    # Compute accuracy based on answer lines
    gold_answer = doc["answer_index"] + 1
    if str(gold_answer) in answer_lines[-1]:
        return {"acc": 1}
    
    return {"acc": 0}


def process_results_ko(doc, results):
    continuation = results[0]

    lines = continuation.split("\n")
    if len(lines) == 0:
        return {"acc": 0}

    answer_lines = [l for l in lines if has_answer(l)]

    if not answer_lines:
        return {"acc": 0}

    if "answer" in doc:
        gold_answer = doc["answer"] + 1
    elif "answer_index" in doc:
        gold_answer = doc["answer_index"] + 1
    else:
        raise KeyError("문서에 'answer' 또는 'answer_index' 키가 없습니다.")

    final_line = answer_lines[-1].strip().replace(" ", "")

    if str(gold_answer) in final_line:
        return {"acc": 1}
    
    return {"acc": 0}


def doc_to_target(doc):
    """Convert a document to a target string."""
    if "reasoning" in doc:
        return doc["reasoning"]
    return doc["answer_index"] + 1

def doc_to_target_ko(doc):
    if "reasoning" in doc:
        return doc["reasoning"]
    return doc["answer"] + 1
    