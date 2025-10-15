import sys
import time
from typing import Dict, Any, List, Callable, Union
from copy import deepcopy
import random
from tqdm import tqdm
from functools import partial


from src import set_seed
from src.madlib.madlib import Madlib
from src.logic_tree.tree import LogicNode, LogicTree, LogicNodeFactType
from src.model import Model
from src.validators import Validator, StructureValidator
from src.utils.constants import NodeTypeConstants, Conjunctions, PromptStep, STORY

# 이 프롬프트는 필요에 따라 덮어쓸 수 있으며, 기본적으로 추론을 생성하는 데 사용됩니다.
_base_completion_prompt_intro_ = f'''
우리는 텍스트 기반 모험 게임 가이드를 만들고 있습니다. 이를 위해 함의 트리를 사용합니다. 
함의 트리는 트리 구조이며, 중간 노드는 자식 노드들에 의해 추론됩니다. 
이 구조는 특정 사실들의 집합에 대한 자연어 기반 추론 과정을 형성합니다.

이 트리를 채우기 위해 함의를 완성해야 합니다. 
함의를 완성하는 것은 함의 트리의 하나의 하위 트리를 채우는 것과 유사합니다. 
이 단계를 채우려면 반드시 주어진 구조를 따라야 합니다.

"{NodeTypeConstants.FACT_FROM_STORY}"은 {STORY}를 작성할 때 명확하게 표현되는 사실입니다.
"{NodeTypeConstants.COMMONSENSE_KNOWLEDGE}"은 대부분의 사람들이 사실이라고 동의하지만 명시적으로 언급할 필요가 없는 정보입니다.

각 단계의 모든 사실은 부모 노드의 주요 사실을 추론할 수 있도록 결합되어야 합니다.

어떤 사실도 현재 트리와 모순될 수 없습니다.

항상 제공된 함의 단계의 정확한 구조를 따르세요. 
"{NodeTypeConstants.FACT_FROM_STORY}"과 "{NodeTypeConstants.COMMONSENSE_KNOWLEDGE}"의 개수를 동일하게 맞추고, 순서도 유지하세요.
'''.strip()


def __create_completion_prompt__(
        example_trees: List[LogicTree],
        example_nodes: List[LogicNode],
        example_descriptions: List[str],
        intro=_base_completion_prompt_intro_,
        pad_char: str = '> ',
        because_clause_after: int = -1,
        because_clause: str = '왜냐하면, ',
        use_complex_facts: bool = False
) -> Callable[[LogicTree, LogicNode, str], str]:
    """
    추론을 요청할 때마다 현재 트리의 상태와 생성할 함의를 전달해야 합니다. 
    이 과정이 재귀적으로 호출되므로, 보다 간편하게 사용할 수 있도록 큰 프롬프트를 
    미리 정의한 후, 나중에 일부 매개변수만 전달하여 사용할 수 있도록 partial 함수로 감싸줍니다.

    이 함수는 그 목적을 수행합니다.

    모든 ICL(인-컨텍스트 학습) 관련 인자는 zip으로 묶여야 하므로 동일한 길이를 가져야 합니다.

    프롬프트 형식은 다음과 같습니다:

    "
    {소개 프롬프트}

    예제를 확인하세요.

    시나리오:
    {ICL 예제 설명 1}

    현재 트리:
    {ICL 트리 1}

    완성해야 할 함의 단계:
    {ICL 노드 1}

    출력:
    {ICL 노드 1의 자식이 포함된 최종 출력}

    ... 모든 ICL 예제 반복 ...

    이제 당신의 차례입니다.

    시나리오:
    {현재 시나리오}

    현재 트리:
    {현재 트리}

    완성해야 할 함의 단계:
    {현재 노드}

    출력:
    "

    :param example_trees: ICL 트리 목록.
    :param example_nodes: ICL에서 완성해야 할 노드 목록.
    :param example_descriptions: ICL에서 추론을 안내하는 설명 목록.
    :param intro: 소개 프롬프트(기본값: _base_completion_prompt_intro_).
    :param pad_char: 트리 출력 시 깊이를 나타내는 문자 (기본값: '> ').
    :param because_clause_after: 특정 깊이 이후에 "왜냐하면" 문구를 추가할지 여부.
    :param because_clause: "왜냐하면" 문구. 예를 들어, "노드 값 + '왜냐하면, '" 형태로 사용됨.
    :param use_complex_facts: "복합 사실"을 사용할지 여부. 자식 노드를 가진 노드에 대해 "복합 사실"을 사용하면 LLM이 나중에 세분화할 수 있도록 도와줌.
    :return: 나중에 특정 매개변수를 받아 최종 프롬프트를 생성하는 partial 함수.
    """
    def node_str(node, pad_char: str = '> ', completed: bool = False, because_clause_after: int = -1, because_clause: str = f'{Conjunctions.BECAUSE} ', use_complex_facts: bool = False):
        def node_line(_node, level, pad_char, tailing_clause: bool = True, because_clause_after: int = -1, because_clause: str = f'{Conjunctions.BECAUSE} '):
            return f'{pad_char * level}{_node.value}' + (('' if _node.value.lower().endswith(Conjunctions.UNLESS) or level <= because_clause_after else f' {because_clause}') if tailing_clause else '')

        parents = []
        p = node.parent
        while p is not None:
            parents.append(p)
            p = p.parent
        parents = parents[::-1]

        estep = []
        for level, p in enumerate(parents):
            estep.append(node_line(p, level, pad_char, because_clause_after=because_clause_after, because_clause=because_clause))

        estep.append(node_line(node, len(parents), pad_char, because_clause_after=because_clause_after, because_clause=because_clause))
        for c in node.children:
            child_str = f'{pad_char * (len(parents) + 1)}'
            if c.fact_type == LogicNodeFactType.EXPLICIT:
                if completed:
                    child_str += node_line(c, 0, pad_char, tailing_clause=False, because_clause_after=because_clause_after, because_clause=because_clause) + ' | '
                if use_complex_facts and len(c.children) > 0:
                    child_str += NodeTypeConstants.COMPLEX_FACT
                else:
                    child_str += NodeTypeConstants.FACT_FROM_STORY
            else:
                if completed:
                    child_str += node_line(c, 0, pad_char, tailing_clause=False, because_clause_after=because_clause_after, because_clause=because_clause) + ' | '
                child_str += NodeTypeConstants.COMMONSENSE_KNOWLEDGE
            estep.append(child_str)
        return "\n".join(estep).strip()

    ex_strs = []
    for (example_tree, example_node, example_description) in zip(example_trees, example_nodes, example_descriptions):
        example_tree_str = example_tree.print_for_gpt(pad_space=1, pad_char=pad_char)
        example_node_str = node_str(example_node, because_clause_after=because_clause_after, use_complex_facts=use_complex_facts, because_clause=because_clause)
        example_completion_str = node_str(example_node, completed=True, because_clause_after=because_clause_after, use_complex_facts=use_complex_facts, because_clause=because_clause)

        ex_strs.append(f'''
{PromptStep.SCENARIO}
{example_description}

{PromptStep.CURRENT_TREE}
{example_tree_str}

{PromptStep.ENTAILMENT_STEP_TO_COMPLETE}
{example_node_str}

출력:
{example_completion_str}

            '''.strip())

    ex_str = "\n또 다른 예제입니다.\n\n".join(ex_strs)

    def prompt(
        tree: LogicTree,
        node: LogicNode,
        description: str,
        ex_str: str,
        _intro: str,
        pad_char: str = '> '
    ):
        p = f'''
{_intro}

예제를 확인하세요.

{ex_str}

이제 당신의 차례입니다.

{PromptStep.SCENARIO}
{description}

{PromptStep.CURRENT_TREE}
{tree.print_for_gpt(pad_char=pad_char, pad_space=1, print_only_nodes_with_value=True)}

{PromptStep.ENTAILMENT_STEP_TO_COMPLETE}
{node_str(node, pad_char=pad_char, because_clause_after=because_clause_after, use_complex_facts=use_complex_facts, because_clause=because_clause).strip()}

'{PromptStep.ENTAILMENT_STEP_TO_COMPLETE_STR}'은 '현재 트리'의 일부입니다. '{PromptStep.ENTAILMENT_STEP_TO_COMPLETE_STR}'에 대해서만, 정확한 개수의 '{NodeTypeConstants.FACT_FROM_STORY}'과 '{NodeTypeConstants.COMMONSENSE_KNOWLEDGE}'를 생성하세요.
출력:
            '''.strip()
        return p

    return partial(prompt, ex_str=ex_str, _intro=intro, pad_char=pad_char)

class DatasetBuilder:
    """
    MuSR 도메인 데이터셋을 생성하는 데 도움을 주는 클래스입니다. 
    
    대부분의 데이터셋은 이 클래스를 상속하고 메서드를 수정하여 사용하게 됩니다.
    (각 도메인이 서로 다른 방식으로 데이터를 처리하기 때문입니다). 
    
    그러나, 재귀적 트리 확장과 같은 상위 수준의 기능은 모든 도메인에서 공통적으로 사용됩니다.
    
    이 클래스는 일종의 유용한 기능 집합으로 생각할 수 있습니다.
    """

    def build_madlib(
            self,
            model: Model,
            things_to_create_names: List[str],
            things_to_create_description: List[str],
            examples_of_things_to_create: List[List[str]],
            max_n_creations: int = 200
    ) -> Madlib:
        """
        특정 항목 목록을 미리 정의된 JSON 파일로 보유하고 있지 않은 경우, 
        이 함수를 사용하여 LLM을 호출하고 원하는 항목을 샘플링할 수 있습니다. 
        
        이 함수는 LLM에서 항목을 샘플링한 후 Madlib 클래스에 저장하여 반환합니다.

        :param model: 항목을 샘플링할 LLM.
        :param things_to_create_names: Madlib 클래스에서 항목을 저장할 이름.
        :param things_to_create_description: 샘플링할 항목의 설명 (프롬프트에서 사용됨).
        :param examples_of_things_to_create: 샘플링할 항목에 대한 예제 (프롬프트에서 사용되며, 최종 리스트에도 포함됨).
        :param max_n_creations: 샘플링할 항목의 최대 개수 (프롬프트에서 사용되며, 하드 제한은 아님).
        :return: 생성된 Madlib 객체.
        """

        assert len(things_to_create_description) == len(things_to_create_names) == len(examples_of_things_to_create), \
            '생성할 항목의 이름, 설명, 예제 목록의 길이가 동일해야 합니다.'
        madlib_data = {}

        max_batchsize = 30
        remaining = max_n_creations
        for name, desc, examples in zip(things_to_create_names, things_to_create_description, examples_of_things_to_create):
            items_result = []
            example_str = '\n'.join(examples)
            while remaining > 0:
                num_requests = min(remaining, max_batchsize)
                prompt = f'{desc}\n\n다음은 몇 가지 예제입니다.\n\n출력:\n{example_str}\n\n이제 당신의 차례입니다. 서로 다른 {num_requests}개 예제를 생성하세요.\n\n출력:'
                output, _ = self.inference(prompt, model, temperature=0.6)
                items = output.split('\n')
                items = [x for x in items if x.strip() != '']
                items_result.extend(items)
                remaining -= len(items)
            assert len(items) > 0, '항목을 샘플링하지 못했습니다.'
            madlib_data[name] = items_result
            # madlib_data[name].extend(examples)

        
        return Madlib(madlib_data)

    def sample_madlib(
            self,
            madlib: Madlib,
            sampled_items: List[Union[str, List[str]]],
            description_string_format: str = None,
            previously_sampled: List[str] = None,
            sampled_item_names: List[str] = None,
            n_samples: int = 1
    ):
        """
        Madlib에서 항목을 샘플링하는 복잡한 방법.

        "sampled_items" 목록을 통해 샘플링할 항목을 지정할 수 있습니다. 
        또한, 여러 항목을 무작위로 선택하여 샘플링할 수도 있습니다.

        예를 들어:
        sampled_items=['motive', ['female_names', 'male_names']]
        → female_names 또는 male_names 중 하나를 무작위로 선택하여 샘플링합니다.

        특정 샘플링 결과에 따라 추가 항목을 샘플링하려면, 콤마로 구분된 문자열을 사용할 수 있습니다.
        sampled_items=['motive', ['female_names,female_relationships', 'male_names,male_relationships']]

        :param madlib: Madlib에서 샘플링할 데이터.
        :param sampled_items: 샘플링할 항목 목록.
        :param description_string_format: 반환할 문자열의 형식 (예: "{name}의 동기는 {motive}").
        :param previously_sampled: 이전에 샘플링된 항목 목록 (중복 방지용).
        :param sampled_item_names: 샘플링된 항목의 이름을 변경 (예: "motive_1", "motive_2").
        :param n_samples: 샘플링할 개수.
        :return: 문자열 리스트, 샘플링된 항목 딕셔너리 리스트, 업데이트된 previously_sampled 목록.
        """
        if description_string_format is None:
            description_string_format = ''
        if previously_sampled is None:
            previously_sampled = []

        out_strings = []
        out_dicts = []
        n = 0
        while n < n_samples:
            out_string = deepcopy(description_string_format)
            sample = []
            out_dict = {}
            for idx, i in enumerate([y for x in sampled_items for y in (x.split(',') if isinstance(x, str) else random.sample(x, 1)[0].split(','))]):
                ignore_list = [x[idx] for x in previously_sampled]
                val = madlib.sample(i, [])[0]
                sample.append(val)

                if sampled_item_names is None:
                    out_string = out_string.replace('{'+i+'}', val)
                    out_dict[i] = val
                else:
                    out_string = out_string.replace('{'+sampled_item_names[idx]+'}', val)
                    out_dict[sampled_item_names[idx]] = val
            print(sample)
            if sample in previously_sampled:
                continue
            n += 1
            previously_sampled.append(sample)
            out_strings.append(out_string)
            out_dicts.append(out_dict)
        return out_strings, out_dicts, previously_sampled

    def build_structure(
            self,
            depth: int = 4,
            bf_factor: Dict[int, float] = None,
            chance_to_prune_all: float = 0.45,
            chance_to_prune: float = 0.5,
            root_nodes: List[LogicNode] = None
    ) -> LogicTree:
        """
        논리 트리(LogicTree)를 생성하는 함수.

        매개변수를 줄여 간단하게 사용할 수 있도록 구성됨. 자세한 사항은 LogicTree 클래스를 참조하세요.

        :return: 생성된 LogicTree.
        """

        return LogicTree(
                chance_of_or=0.0,
                chance_of_cs_fact=0.0,
                depth=depth,
                chance_to_prune=chance_to_prune,
                chance_to_prune_all=chance_to_prune_all,
                bf_factor=bf_factor,
                enforce_cs_fact_per_level=True,
                deduction_type_sample_rate=None,
                root_structure=root_nodes
            )

    def create_completion_prompt(
            self,
            example_trees: List[LogicTree],
            example_node_completions: List[LogicNode],
            example_descriptions: List[str],
            intro: str = _base_completion_prompt_intro_,
            pad_char: str = '> ',
            because_clause_after: int = -1,
            because_clause: str = '왜냐하면, ',
            use_complex_facts: bool = False
    ):
        """__create_completion_prompt__에 대한 래퍼 함수. 더 자세한 내용은 __create_completion_prompt__를 참조하세요."""
        return __create_completion_prompt__(example_trees, example_node_completions, example_descriptions, intro=intro, pad_char=pad_char, because_clause_after=because_clause_after, because_clause=because_clause, use_complex_facts=use_complex_facts)

    def complete_structure(
            self,
            _tree: LogicTree,
            model: Model,
            description: str,
            completion_prompt_fn: Callable[[LogicTree, LogicNode, str], str],
            max_retries_on_error: int = 1,
            inplace: bool = False,
            retry_model: Model = None,
            progress_bar: bool = False,
            test_prompt: bool = False,
            use_iterative_complete_v2: bool = False,
            validators: List[Validator] = (StructureValidator())
    ) -> LogicTree:
        """
        재귀적 추론 트리 확장(Recursive Reasoning Tree Expansion) 알고리즘의 시작점입니다.

        이 함수는 전체 구조(자식 노드가 모두 생성되고 populate/prune 함수가 호출됨)를 가진 트리를 입력으로 받습니다.
        그러나 일부 노드는 여전히 내용이 없을 수 있습니다. 이 함수는 트리를 재귀적으로 순회하며 모든 노드의 내용을 채웁니다.

        :param _tree: 채워야 할 템플릿 트리.
        :param model: 추론을 수행할 모델.
        :param description: 스토리 또는 예제와 관련된 설명.
        :param completion_prompt_fn: 현재 생성 중인 함의 단계를 입력받아 적절한 추론을 생성하는 프롬프트를 반환하는 함수.
        :param max_retries_on_error: 검증 실패 시 재시도 횟수.
        :param inplace: 트리를 직접 수정할지 여부.
        :param retry_model: 재시도 시 사용할 모델 (예: GPT-3.5에서 실패하면 GPT-4를 사용).
        :param progress_bar: 진행 상황을 TQDM 바 형태로 표시할지 여부.
        :param test_prompt: 첫 번째 프롬프트를 출력한 후 프로그램을 종료 (디버깅용).
        :param use_iterative_complete_v2: 구조 검증 외의 추가 검증을 사용할지 여부.
        :param validators: 사용할 검증기 리스트.
        :return: 완성된 LogicTree.
        """

        def get_num_steps(node):
            """트리에서 채워야 할 단계의 총 개수를 계산합니다."""
            return (1 if any([x.value == '' for x in node.children]) else 0) + sum(
                [get_num_steps(x) for x in node.children])
        
        pbar = tqdm(total=sum([get_num_steps(x) for x in _tree.nodes]), desc='구조 채우기 진행 중...', disable=not progress_bar)

        if retry_model is None:
            retry_model = model

        if not inplace:
            tree = deepcopy(_tree)
        else:
            tree = _tree

        def iteratively_complete(
                description: str,
                tree: LogicTree,
                node: LogicNode,
                model: Model,
                retry_model: Model,
                completion_prompt_fn,
                pbar,
                pad_char='> ',
                max_retries_on_error: int = 1,
                test_prompt: bool = False
        ):
            """재귀적 추론 트리 확장 알고리즘 v1 (검증 미포함, 사용되지 않음)"""
            children = node.children
            if not tree.valid:
                # 트리가 유효하지 않으면 더 이상 진행하지 않음
                return

            if any([x.value == '' for x in children]):
                prompt = completion_prompt_fn(tree, node, description)

                if test_prompt:
                    print(prompt)
                    sys.exit(0)

                raw = model.inference(prompt)
                output = raw["choices"][0]['message']['content']

                def parse_out(output):
                    """출력에서 명시적 사실과 상식적 지식을 파싱합니다."""
                    facts_from_story = []
                    cs_knowledge = []

                    for l in output.split('\n'):
                        val = '|'.join(l.replace(f'{pad_char}', '').split('|')[:-1])
                        if val == node.value:
                            continue

                        if f'| {NodeTypeConstants.FACT_FROM_STORY}' in l or f'| {NodeTypeConstants.COMPLEX_FACT}' in l:
                            if val not in facts_from_story and val not in cs_knowledge:
                                facts_from_story.append(val)
                        elif f'| {NodeTypeConstants.COMMONSENSE_KNOWLEDGE}' in l:
                            if val not in facts_from_story and val not in cs_knowledge:
                                cs_knowledge.append(val)
                    return facts_from_story, cs_knowledge

                facts_from_story, cs_knowledge = parse_out(output)

                retry_idx = 0
                while retry_idx <= max_retries_on_error and len(facts_from_story) + len(cs_knowledge) != len(node.children):
                    retry_idx += 1

                    prompt += f'이전 출력이 잘못되었습니다:\n{output}\n\n이 구조와 일치하지 않거나, 동일한 사실을 두 번 포함했습니다. 이번에는 정확한 개수의 "{NodeTypeConstants.FACT_FROM_STORY}"과 "{NodeTypeConstants.COMMONSENSE_KNOWLEDGE}"을 제공하고, 중복되지 않도록 생성하세요.\n\n출력:'

                    if retry_idx == 1:
                        raw = model.inference(prompt)
                    else:
                        raw = retry_model.inference(prompt)

                    output = raw["choices"][0]['message']['content']
                    facts_from_story, cs_knowledge = parse_out(output)

                try:
                    for c in node.children:
                        if c.fact_type == LogicNodeFactType.COMMONSENSE:
                            c.value = cs_knowledge.pop()
                        elif c.fact_type == LogicNodeFactType.EXPLICIT:
                            c.value = facts_from_story.pop()
                except Exception as e:
                    print('오류 발생 (브랜치 삭제): ' + str(e))
                    node.children = []
                    tree.valid = False
                    return

                pbar.update(1)
            for c in children:
                iteratively_complete(description, tree, c, model, retry_model, completion_prompt_fn, pbar, max_retries_on_error=max_retries_on_error, test_prompt=test_prompt)

        if use_iterative_complete_v2:
            [self.iteratively_complete_v2(description, tree, x, model, retry_model, completion_prompt_fn, pbar, max_retries_on_error=max_retries_on_error, test_prompt=test_prompt, validators=validators) for x in tree.nodes]
        else:
            [iteratively_complete(description, tree, x, model, retry_model, completion_prompt_fn, pbar, max_retries_on_error=max_retries_on_error, test_prompt=test_prompt) for x in tree.nodes]

        return tree

    def inference(
            self,
            prompt: str,
            model: Model,
            temperature: float = None
    ) -> [str, Any]:
        """모델에서 추론을 실행하는 헬퍼 함수"""
        if temperature:
            raw = model.inference(prompt, temperature=temperature)
        else:
            raw = model.inference(prompt)
        output = raw["choices"][0]['message']['content']

        return output, raw

    def create_dataset_question_object(
            self,
            context: str,
            questions: List[str],
            answers: List[int],
            choices: List[List[str]],
            intermediate_trees: List[List[LogicTree]],
            intermediate_data: List[List[Any]] = None,
    ):
        """
        데이터셋의 형식을 일관되게 유지합니다.

        :param context: 질문을 위한 배경 정보 (예: 실제 사건 개요)
        :param questions: 배경 정보에 대해 묻는 질문 목록
        :param answers: 각 질문에 대한 정답 (정답 선택지의 인덱스)
        :param choices: 각 질문에 대한 선택지 목록 (리스트의 리스트)
        :param intermediate_trees: 중간 추론 트리 (각 질문의 선택지별 하나씩 존재)
        :param intermediate_data: 중간 데이터 (각 질문에 대한 데이터 목록)
        """

        if intermediate_data is None:
            intermediate_data = [[None] * len(intermediate_trees)]
        intermediate_trees = [[x.to_json() if isinstance(x, LogicTree) else x for x in y] for y in intermediate_trees]
        intermediate_data = [[x.to_json() if isinstance(x, LogicTree) else x for x in y] for y in intermediate_data]
        questions = [{'question': x, 'answer': y, 'choices': z, 'intermediate_trees': i, 'intermediate_data': j} for x, y, z, i, j in zip(questions, answers, choices, intermediate_trees, intermediate_data)]

        return {
            'context': context,
            'questions': questions,
        }

    def iteratively_complete_v2(
            self,
            description: str,
            tree: LogicTree,
            node: LogicNode,
            model: Model,
            retry_model: Model,
            completion_prompt_fn,
            pbar,
            pad_char='> ',
            max_retries_on_error: int = 1,
            test_prompt: bool = False,
            validators: List[Validator] = (StructureValidator()),
    ):
        """재귀적 추론 트리 확장 알고리즘 v2"""
        children = node.children

        if not tree.valid:
            # 트리가 유효하지 않으면 더 이상 진행하지 않음
            return

        if any([x.value == '' for x in children]):
            # 현재 노드의 자식 노드 중 값이 없는 것이 있다면, 추론을 생성해야 함

            def parse_out(output):
                """모델 출력을 파싱하여 '스토리에서 추출된 사실'과 '상식적 지식'을 추출"""
                facts_from_story = []
                cs_knowledge = []

                for l in output.split('\n'):
                    val = '|'.join(l.replace(f'{pad_char}', '').split('|')[:-1])
                    if val == node.value:
                        continue

                    if f'| {NodeTypeConstants.FACT_FROM_STORY}' in l or f'| {NodeTypeConstants.COMPLEX_FACT}' in l:
                        if val not in facts_from_story and val not in cs_knowledge:
                            facts_from_story.append(val)
                    elif f'| {NodeTypeConstants.COMMONSENSE_KNOWLEDGE}' in l:
                        if val not in facts_from_story and val not in cs_knowledge:
                            cs_knowledge.append(val)
                return facts_from_story, cs_knowledge

            prompt = completion_prompt_fn(tree, node, description)

            if test_prompt:
                print(prompt)
                sys.exit(0)

            facts_from_story = []
            cs_knowledge = []

            retry_idx = 0

            # 검증기를 통해 생성된 결과를 검증
            all_valid = True
            while retry_idx <= max_retries_on_error:
                all_valid = True
                raw = model.inference(prompt)
                output = raw["choices"][0]['message']['content']

                facts_from_story, cs_knowledge = parse_out(output)

                for v in validators:
                    valid, retry_prompt = v(node, facts_from_story, cs_knowledge, output)
                    if not valid:
                        # 실패 시, 검증기의 재시도 프롬프트를 추가한 후 다시 요청
                        prompt_parts = prompt.split(PromptStep.ENTAILMENT_STEP_TO_COMPLETE)

                        prompt = PromptStep.ENTAILMENT_STEP_TO_COMPLETE.join(prompt_parts[:-1]) + f'\n\n{retry_prompt}\n\n{PromptStep.ENTAILMENT_STEP_TO_COMPLETE}\n{prompt_parts[-1].strip()}'
                        all_valid = False
                        break
                if all_valid:
                    break

                retry_idx += 1

            if not all_valid:
                print('오류 발생: 검증 실패 (브랜치 삭제)')
                node.children = []
                tree.valid = False
                return
            else:
                try:
                    for c in node.children:
                        if c.fact_type == LogicNodeFactType.COMMONSENSE:
                            c.value = cs_knowledge.pop()
                        elif c.fact_type == LogicNodeFactType.EXPLICIT:
                            c.value = facts_from_story.pop()
                except Exception as e:
                    print('오류 발생 (브랜치 삭제): ' + str(e))
                    node.children = []
                    tree.valid = False
                    return

            pbar.update(1)
        for c in children:
            self.iteratively_complete_v2(
                description,
                tree,
                c,
                model,
                retry_model,
                completion_prompt_fn,
                pbar,
                max_retries_on_error=max_retries_on_error,
                test_prompt=test_prompt,
                validators=validators
            )
