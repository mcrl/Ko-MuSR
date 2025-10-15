from tqdm import tqdm
import sys
from typing import List, Dict, Union, Callable, Tuple
from functools import partial
import random
from copy import deepcopy


from src.madlib.madlib import Madlib
from src.logic_tree.tree import LogicTree, LogicNode, LogicNodeFactType
from src.dataset_builder import DatasetBuilder
from src.model.openai import Model
from src.validators import StructureValidator, Validator, ForbiddenTextValidator, ModelValidator
from src.utils.constants import MysteryConstants, NodeTypeConstants, PromptStep, STORY

# 이 스크립트는 dataset_builder.py에 정의된 _base_completion_prompt_intro_를 덮어쓰기 위한 목적으로 사용됩니다.
# 특히, 용의자의 수단(means), 동기(motive), 기회(opportunity)를 구성하는 브랜치를 생성할 때 사용됩니다.
entailment_step_to_complete_str = PromptStep.remove_colon(PromptStep.ENTAILMENT_STEP_TO_COMPLETE)
_mm_completion_prompt_intro_ = f'''
당신의 임무는 예시에 제시된 방식으로 {STORY}를 위한 논리 트리를 생성하는 것입니다. 
이 트리에서는 각 사실이 자신의 자식 노드로부터 논리적으로 도출되어야 합니다. 
이미 이름이 있는 사실(루트 노드 등)이 있을 경우, 새롭게 덮어쓰지 말아야 합니다.

이야기의 종류:

우리는 살인 미스터리를 만들고 있습니다. 살인 미스터리에는 용의자가 살인을 저지를 수 있는 '{MysteryConstants.MEANS}, {MysteryConstants.MOTIVE}, {MysteryConstants.OPPORTUNITY}'가 뒤얽힌 복잡한 증거망이 필요합니다. 
{STORY}는 형사의 시점에서 쓰여야 합니다. 증거를 모으는 과정은 보통의 수사 방식(심문, 대화 엿듣기, 범죄 기록 조회, 우편물·쓰레기 조사 등)을 통해 이뤄집니다.

1. 트리에 등장하는 각 사실은 자식 노드로부터 논리적으로 도출되어야 합니다.
2. 모든 "{NodeTypeConstants.FACT_FROM_STORY}" 노드와 "{NodeTypeConstants.COMMONSENSE}" 노드는, 결과적으로 만들어지는 추론에 밀접하게 관련되어야 합니다.
3. 각 루트 사실(root fact)에는 출처가 표시됩니다("{NodeTypeConstants.FACT_FROM_STORY}" 혹은 "{NodeTypeConstants.COMMONSENSE}").
4. "{NodeTypeConstants.FACT_FROM_STORY}"은 이야기 속 캐릭터, 장소, 물건에 대한 구체적 진술이어야 합니다.
5. "{NodeTypeConstants.COMMONSENSE}"은 대부분의 사람이 알고 동의할 만한 사실이어야 합니다. 이야기 속 특정 인물이나 사건을 직접적으로 언급하면 안 됩니다.
6. "{NodeTypeConstants.COMMONSENSE}"은 형식상 '규칙(rule)' 역할을 하며, 형제 노드의 사실을 결합해 상위 사실을 도출하는 용도로 사용됩니다.
7. 생성되는 트리는 '{entailment_step_to_complete_str}'의 트리 구조와 일치해야 합니다.
8. '{entailment_step_to_complete_str}' 트리의 "{NodeTypeConstants.FACT_FROM_STORY}" 요청과 "{NodeTypeConstants.COMMONSENSE}"에 대해 한 개씩의 노드를 생성해야 합니다.
9. '{entailment_step_to_complete_str}' 트리에서 요청되지 않은 "{NodeTypeConstants.FACT_FROM_STORY}"이나 "{NodeTypeConstants.COMMONSENSE}" 노드를 생성하지 마세요.

용의자가 '{MysteryConstants.MEANS}'을 가졌다는 것은 살해 도구에 접근할 수 있음을 의미합니다.
용의자가 '{MysteryConstants.MOTIVE}'를 가졌다는 것은 그(또는 그녀)가 피해자를 죽일 이유가 있음을 의미합니다.
용의자가 '{MysteryConstants.OPPORTUNITY}'를 가졌다는 것은 범행 현장에 있었음을 의미합니다.

설령 황당하거나 비현실적인 설정의 부모 노드 사실이라도, 그 사실을 유지하고 논리 트리를 구성해야 합니다.
'''.strip()

# 이 스크립트는 dataset_builder.py에 정의된 _base_completion_prompt_intro_를 덮어쓰기 위한 목적으로 사용됩니다.
# 특히, 각 용의자에 대한 의심스러운 사실(suspicious facts) 트리를 생성할 때 사용됩니다.
_mm_suspicious_prompt_intro_ = f'''
당신의 임무는 예시에 제시된 방식으로 {STORY}를 위한 논리 트리를 생성하는 것입니다. 
이 트리에서는 각 사실이 자신의 자식 노드로부터 논리적으로 도출되어야 합니다. 
이미 이름이 있는 사실(루트 노드 등)이 있을 경우, 새롭게 덮어쓰지 말아야 합니다.

이야기의 종류:

우리는 살인 미스터리를 만들고 있습니다. 살인 미스터리에는 '정말 의심스러워 보이지만 실제로는 범인임을 입증하지 못하는' 헛다리 같은 요소도 필요합니다.

여기서는 헛다리를 생성합니다. 이 요소들은 의심스러워 보이지만, 실제로는 용의자가 유죄임을 입증하는 결정적 단서가 될 수 없습니다. 
당신은 용의자가 유죄라는 증거를 제시하지 않을 것입니다.

1. 트리에 등장하는 각 사실은 자식 노드로부터 논리적으로 도출되어야 합니다.
2. 모든 "{NodeTypeConstants.FACT_FROM_STORY}" 노드와 "{NodeTypeConstants.COMMONSENSE}" 노드는, 결과적으로 만들어지는 추론(부모 사실)에 밀접하게 관련되어야 합니다.
3. 각 루트 사실에는 출처가 표시됩니다(사실 혹은 상식).
4. "{NodeTypeConstants.FACT_FROM_STORY}"은 이야기 속 캐릭터, 장소, 물건에 대한 구체적 진술이어야 합니다.
5. "{NodeTypeConstants.COMMONSENSE}"은 대부분의 사람이 알고 동의할 만한 사실이어야 합니다. 이야기 속 특정 인물이나 사건을 직접적으로 언급하면 안 됩니다.
6. "{NodeTypeConstants.COMMONSENSE}"은 형식상 '규칙' 역할을 하며, 형제 노드의 사실을 결합해 상위 사실을 도출하는 용도로 사용됩니다.
7. 생성되는 트리는 '{entailment_step_to_complete_str}'의 트리 구조와 일치해야 합니다.
8. '{entailment_step_to_complete_str}' 트리의 "{NodeTypeConstants.FACT_FROM_STORY}" 요청과 "{NodeTypeConstants.COMMONSENSE}"에 대해 한 개씩의 노드를 생성해야 합니다.
9. '{entailment_step_to_complete_str}' 트리에서 요청되지 않은 "{NodeTypeConstants.FACT_FROM_STORY}"이나 "{NodeTypeConstants.COMMONSENSE}" 노드를 생성하지 마세요.
'''.strip()

"""
의심스러운 사실을 생성하기 위한 ICL 예시입니다. 
이는 수단, 동기, 기회와 직접적인 관련이 없는, 단순히 의심만 끌어내는 정보들이어야 합니다.
"""
cLogicNode = partial(LogicNode, fact_type=LogicNodeFactType.COMMONSENSE)

example1_description = '민수와 예림이는 노래방에 있다.'
example1_tree = LogicTree(
    nodes=[
        LogicNode('민수와 예림이는 노래방에 있다.', [
            LogicNode('오프닝 장면', [
                LogicNode('민수는 무대에 있는 마이크를 보았다.'),
                LogicNode('예림이는 무대에 있는 마이크를 보았다.'),
                LogicNode('민수는 바에 있는 맥주를 보았다.'),
                LogicNode('예림이는 바에 있는 맥주를 보았다.')
            ]),
            LogicNode('민수는 맥주를 테이블로 옮긴다.', [
                LogicNode('에림이는 맥주가 테이블로 옮겨지는 것을 보지 못했다.', [
                    LogicNode('예림이는 테이블에서 반대쪽을 보고 있었다.', [
                        LogicNode('예림이는 다른 손님과 대화 중이었다.'),
                        LogicNode('다른 손님은 테이블 쪽을 보고 있었다.'),
                        cLogicNode('보통 사람들은 서로를 바라보며 대화하기 때문에, 한 사람이 한 방향을 보고 있으면 다른 사람은 반대 방향을 보고 있게 된다.')
                    ]),
                    cLogicNode('만약 누군가 다른 무언가에서 반대 방향을 보고 있다면, 그 무언가 주변에서 일어나는 일을 볼 수 없다.')
                ])
            ]),
            LogicNode('예림이는 마이크를 테이블로 옮긴다', [
                LogicNode('예림이는 마이크를 옮길 때 테이블에 있는 맥주를 보았다.'),
                LogicNode('민수는 마이크가 테이블로 옮겨지는 것을 보았다.', [
                    cLogicNode("민수는 테이블에서 맥주를 마시고 있었다."),
                    LogicNode("어떤 일이 사람 바로 옆에서 일어나면, 그 사람은 보통 그 상황을 볼 수 있다.")
                ])
            ]),
            LogicNode('예림이는 맥주를 쓰레기통으로 옮긴다.', [
                LogicNode('민수는 맥주가 쓰레기통으로 옮겨지는 것을 보지 못했다.')
            ])
        ])

    ], prune=False, populate=False
)
example1_node_completion_tree =  LogicNode('민수와 예림이는 노래방에 있다.', [
    LogicNode('예림이는 맥주를 쓰레기통으로 옮긴다.', [
        LogicNode('민수는 맥주가 쓰레기통으로 옮겨지는 것을 보지 못했다.', [
            LogicNode('예림이는 민수를 속여서 "저쪽"을 보게 했다.'),
            LogicNode('예림이는 민수에게 쓰레기통의 반대 방향을 가리켰다.'),
            cLogicNode('누군가를 속여서 다른 곳을 보게 만들면, 반대쪽에서 일어나는 일을 볼 수 없다.')
        ])
    ])
])



example2_description = '당신의 개가 방금 이웃의 마당에 똥을 쌌다. 이웃은 당신을 노려보며 다가온다... 그는 이렇게 말한다: "이봐요! 지금 당신 개가 내 예쁜 마당에다 무슨 짓을 하게 두는 거죠!"'
example2_tree = LogicTree(
    nodes=[
        LogicNode('이웃의 코를 정면으로 주먹질한다.', [
            LogicNode('멋져 보일 것이고, 이것은 장점이다.', [
                LogicNode('당신이 사는 곳에서는 싸움을 멋지다고 생각한다.'),
                LogicNode('당신은 싸움을 하게 될 것이다.'),
                LogicNode('사람들이 멋지다고 생각하는 행동을 하면, 그 행동을 한 사람도 멋지게 여겨진다.',
                          fact_type=LogicNodeFactType.COMMONSENSE)
            ]),
            LogicNode('멋져 보일 것이지만, 상황에 따라 달라질 수 있다...', [])
        ]),
        LogicNode('이웃에게 "정말 죄송합니다, 훈련 중인데 잘 안되네요."라고 말한다.'),
        LogicNode('이웃이 당신을 해칠 것 같아서 겁이 난다.'),
        LogicNode('이웃은 당신을 내버려 둘 것이다.')
    ], prune=False, populate=False
)
example2_node_completion_tree = LogicNode('이웃의 코를 정면으로 주먹질한다.',
                                     [LogicNode('멋져 보일 것이지만, 상황에 따라 달라질 수 있다...', [
                                        LogicNode('당신은 이웃을 해칠 수 있다.'),
                                        LogicNode('당신은 이유 없이 공격한 셈이다.'),
                                        cLogicNode('아무런 이유 없이 누군가를 해치는 것은 멋지지 않다.')
                                    ])])

example3_description = f"""
{MysteryConstants.VICTIM}: 준호
{MysteryConstants.PLACE}: 공원 벤치
{MysteryConstants.TOOL}: 헤로인 과다복용
{MysteryConstants.SUSPECT}: 민철
{MysteryConstants.ROLE}: 마약 사용자
{MysteryConstants.MOTIVE}: 공개적인 굴욕
"""
minchul = "민철"
example3_tree = LogicTree(
    nodes=[
        LogicNode(MysteryConstants.is_murderer(minchul), [
            LogicNode(MysteryConstants.has_means(minchul)),
            LogicNode(MysteryConstants.has_motive(minchul)),
            LogicNode(MysteryConstants.has_opportunity(minchul))
        ])
    ],
    prune=False, populate=False
)

example3_node_completion = LogicNode(MysteryConstants.has_means(minchul), [
    LogicNode("민철이는 헤로인에 접근할 수 있다."),
    LogicNode("민철이는 헤로인 과다복용에 필요한 양을 알고 있다."),
    LogicNode(
        "헤로인에 접근할 수 있고, 치명적인 양을 알고 있다면, 피해자에게 의도적으로 치명적인 용량을 투여하여 살인을 저지를 수 있는 수단을 갖춘 것이다.",
        fact_type=LogicNodeFactType.COMMONSENSE
    )
])


sf_example_descriptions = [example1_description, example2_description, example3_description]
sf_example_trees = [example1_tree, example2_tree, example3_tree]
sf_example_node_completions = [example1_node_completion_tree.children[0].children[0], example3_node_completion, example1_node_completion_tree.children[0]]


def create_story_prompt__facts_only(description: str, story_tree: LogicTree, pad_char: str = '> '):
    """
    이 함수는 스토리(장)를 생성할 때 사용하는 프롬프트를 만들어냅니다. 
    트리(story_tree)에서 말단(leaf) 노드에 해당하는 명시적 사실들을 수집하여, 
    용의자를 다루는 장(chapter)을 쓸 때 참고할 리스트로 사용합니다. 
    ICL 예시는 하나만 제시하는 방식으로 구성됩니다.

    :param description: 이야기를 설명하는 문자열
    :param story_tree: 작성 완료된 트리
    :param pad_char: (사용되지 않음)
    :return: 스토리 작성을 위한 프롬프트
    """
    facts = list(sorted(list(set([x.value for x in story_tree.get_facts()]))))
    random.shuffle(facts)
    facts_str = "\n".join([f'- {x}' for x in facts])

    return f"""
우리는 살인 미스터리를 만들고 있습니다. 
살인 미스터리에는 용의자와 관련된 복잡한 증거들이 필요합니다. 
이야기는 형사의 시점에서 쓰여야 합니다. 
형사는 통상적인 수사 방식을 통해 증거를 수집합니다(심문, 대화 엿듣기, 범죄 기록 조사, 우편물·쓰레기 뒤지기 등).

당신에게 사실 목록을 제공할 것이고, 당신은 이 목록에 있는 사실들을 모두 {STORY}에 포함시켜야 합니다. 
절대 유도된 사실(추론된 사실)이나 결론을 언급해서는 안 됩니다. 이야기의 흐름은 이 사실들을 충실히 따라가야 합니다.

다음은 살인 미스터리에서 용의자와 관련된 한 장(chapter)을 작성해야 합니다. 
이 장에서는 살인 사건이나 피해자에 관한 직접적인 언급을 최소화하고, 
형사가 용의자에게 접근하여 수사를 진행하는 과정을 그립니다. 
아래 규칙을 유의하세요:

1. 장의 내용만 작성하고, 제목이나 번호를 달지 마세요. 다른 장과 연속적으로 이어질 수 있도록 간단한 서술만 하세요.
2. 용의자가 '{MysteryConstants.MEANS}을 가졌다', '{MysteryConstants.MOTIVE}를 가졌다', '{MysteryConstants.OPPORTUNITY}가 있다'라고 절대 말하지 마세요.
3. 용의자에게 {MysteryConstants.MEANS}, {MysteryConstants.MOTIVE}, {MysteryConstants.OPPORTUNITY}가 있다고 암시조차 하지 마세요.
4. 유도된 사실(추론된 사실)을 직접적으로 언급하거나 결론짓지 마세요.
5. 용의자를 살인자로 단정 지어서 말하지 마세요. 그건 독자가 추리해야 할 부분입니다.
6. 이야기의 시점은 형사가 현장 조사를 하거나 용의자를 심문하는 등, 일반적으로 합리적인 수사 방식을 통해 단서를 얻는 상황으로 유지하세요.
7. 이야기 속에서 '이 사실은 의심스러워' 같은 멘트를 하지 마세요. 독자가 추론하게 남겨두세요.
8. 이야기 속에서 형사의 이름은 '김철수'로 통일하세요.

주어진 사실(fact) 목록에서 각 사실을 {STORY} 속에 반드시 등장시키고, 
최대 10개의 대사를 포함해 작성하세요.

다음은 예시입니다:

용의자와 범죄 정보
{MysteryConstants.VICTIM}: 유진
{MysteryConstants.PLACE}: 인적이 드문 숲
{MysteryConstants.TOOL}: 칼
{MysteryConstants.SUSPECT}: 지훈
{MysteryConstants.ROLE}: 학교 운동장 관리인
{MysteryConstants.MOTIVE}: 종교적 희생

당신은 김철수 형사입니다.
반드시 포함해야 할 사실들:
- 어떤 목격자는 스파게티 얼굴과 초록색 귀를 가진 누군가를 보았다
- 지훈이는 지역 학교의 잔디 관리인이다
- 지훈이는 근처 주민들에게 페인트칠, 잔디 손질 등 서비스를 제공한다
- 지훈이는 주변 집 중 한 곳을 초록색으로 페인트칠했다
- 지훈이는 어렸을 때 큰 화상을 입었다
- 지훈이의 가족은 여러 세대 동안 이 지역에 살았다
- 이 지역에서는 예전부터 종교적 극단주의자들이 있었고, 모두 신비로운 종교 의식을 치렀다
- 지훈이와 그의 직계 가족들은 모두 잡다한 기술을 가지고 있었다
- 지훈이는 조상과 전통을 존중하는 신념을 갖고 있다
- 유진이는 근처에 있는 새로운 교회에 참여할 생각을 적어두었다
- 유진이의 친구는 유진이가 새로 사귀게 된 불량스러운 무리와 어울리는 것을 걱정했다

출력 예시:

김철수 형사는 범행 현장 사진을 훑어보면서 담배를 길게 한 모금 빨았다. 그는 끔찍한 장면들에 어느 정도 익숙해져 있었지만, 유진이의 사건에는 이상하게 마음이 쏠렸다...

(중략: 예시 {STORY})

---

이제 당신 차례입니다. {STORY} 외에는 아무것도 출력하지 마세요.

용의자와 범죄 정보
{description}

당신은 김철수 형사입니다.

반드시 포함해야 할 사실들:
{facts_str}

출력:
    """.strip()


class MurderMysteryDataset(DatasetBuilder):
    """
    DatasetBuilder를 상속하여, '살인 미스터리'와 관련된 기능을 확장한 클래스입니다.

    이 클래스는 각 용의자에 대해 수단(means), 동기(motive), 기회(opportunity), 
    그리고 의심스러운 사실(하지만 실제로 범죄와 직접적 연관은 없는) 등의 정보를 담은 트리를 생성하고, 
    그 정보를 바탕으로 스토리를 작성하는 과정을 포함합니다.
    """

    def create_suspect_trees(
            self,
            model: Model,
            victim_info: Dict[str, str],
            suspect_infos: List[Dict[str, str]],
            example_completion_trees: List[LogicTree],
            example_completion_nodes: List[LogicNode],
            example_completion_descriptions: List[str],
            depth: int = 4,
            bf_factor: Dict[int, float] = None,
            chance_to_prune_all: float = 0.45,
            chance_to_prune: float = 0.5,
            max_num_of_suspicious_facts: int = 1,
            max_retries_on_error: int = 1,
            retry_model: Model = None,
            progress_bar: bool = False,
            test_completion_prompt: bool = False,
            use_validators: bool = True,
            model_validator_model: Model = None,
            model_validator_early_escape_model: Model = None
    ):
        """
        각 용의자에 대한 트리를 생성합니다. 
        용의자 트리는 '{MysteryConstants.MEANS}(means), 동기(motive), 기회(opportunity)' 각 브랜치와, 
        의심스러운(redd herring) 사실을 담은 브랜치로 구성됩니다.

        :param model: DatasetBuilder에서 사용하는 모델
        :param victim_info: 피해자의 정보(딕셔너리)
        :param suspect_infos: 각 용의자별 정보(딕셔너리 리스트)
        :param example_completion_trees: ICL 예시로 사용할 LogicTree 목록
        :param example_completion_nodes: ICL 예시로 사용할 LogicNode 목록
        :param example_completion_descriptions: ICL 예시로 사용할 시나리오 설명 목록
        :param depth: 트리의 최대 깊이
        :param bf_factor: 각 레벨의 분기 계수(깊이별 노드 분기 조정)
        :param chance_to_prune_all: 전체를 prune할 확률
        :param chance_to_prune: 각 노드를 prune할 확률
        :param max_num_of_suspicious_facts: 의심스러운 사실 브랜치 개수
        :param max_retries_on_error: 에러 발생 시 재시도 횟수
        :param retry_model: 재시도 시 사용할 모델
        :param progress_bar: 진행 상황 표시 여부
        :param test_completion_prompt: 구조 완성 프롬프트 테스트 여부
        :param use_validators: 유효성 검사(Validators) 사용 여부
        :param model_validator_model: ModelValidator에서 사용할 모델
        :param model_validator_early_escape_model: ModelValidator에서 빠른 종료에 사용할 모델
        """

        suspect_trees = []

        victim = victim_info['victim']
        murder_weapon = victim_info['murder_weapon']
        crime_scene = victim_info['crime_scene']

        for suspect_info in suspect_infos:
            suspect_name = suspect_info['suspect']
            motive = suspect_info['motive']
            try:
                description = suspect_info['description']
            except Exception as e:
                print(e)

            # 각 용의자에 대해 먼저 MMO 트리를 생성
            root_node = [
                LogicNode(f'{suspect_name}', [
                    LogicNode(MysteryConstants.has_means(suspect_name)),
                    LogicNode(MysteryConstants.has_opportunity(suspect_name)),
                    LogicNode(MysteryConstants.has_motive(suspect_name)),
                ], frozen=True, prunable=False)
            ]

            validators = [StructureValidator()]
            if use_validators:
                # 살인 미스터리 관련 전용 ForbiddenTextValidator
                validators.append(ForbiddenTextValidator(
                    forbidden_words=[
                        ['수단이 있다', crime_scene],
                        ['수단이 있다', '범행 장소'],
                        ['수단이 있다', '기회'],
                        ['수단이 있다', motive],
                        ['수단이 있다', '동기'],
                        ['기회가 있다', murder_weapon],
                        ['기회가 있다', '살해 도구'],
                        ['기회가 있다', motive],
                        ['기회가 있다', '무기'],
                        ['동기가 있다', crime_scene],
                        ['동기가 있다', '범행 장소'],
                        ['동기가 있다', '동기'],
                        ['동기가 있다', '살해 도구'],
                        ['동기가 있다', murder_weapon],
                        ['동기가 있다', '무기'],
                    ],
                    reason_why=f"우리는 엄격한 논리적 추론 과정을 통해 살인 미스터리를 만들고자 합니다. '{MysteryConstants.MEANS}' 브랜치에서 '{MysteryConstants.OPPORTUNITY}'가 증명되면 이후에 용의자를 무죄로 만들 때 문제가 생길 수 있습니다. 따라서, {MysteryConstants.MEANS}, {MysteryConstants.MOTIVE}, {MysteryConstants.OPPORTUNITY} 각각의 브랜치에서는 해당되는 사실만 다뤄야 합니다."
                ))

                if model_validator_model:
                    validators.extend([
                        ModelValidator(
                            model_validator_model,
                            f"현재 살인 미스터리를 작성 중이며, 특정 용의자의 {MysteryConstants.MOTIVE}만을 증명하는 단계입니다. 다음은 미스터리 설명입니다:\n\n{description}\n\n주어진 추론은 '{MysteryConstants.MEANS}'(살해 도구 접근) 또는 '{MysteryConstants.MEANS}(살해 도구 접근)'을 포함하면 안 됩니다. 혹시 아래의 추론이 '{MysteryConstants.MEANS}(살해 도구 접근)' 또는 '{MysteryConstants.OPPORTUNITY}(범행 장소에 대한 접근)'를 증명하거나 그 증명을 돕는 내용을 담고 있나요?",
                            f"지금은 {MysteryConstants.MOTIVE}만 증명해야 합니다. '{MysteryConstants.MEANS}'이나 '{MysteryConstants.OPPORTUNITY}' 관련 내용은 배제되어야 합니다.",
                            conditional='동기가 있다',
                            early_escape_model=model_validator_early_escape_model
                        ),
                        ModelValidator(
                            model_validator_model,
                            f"현재 살인 미스터리를 작성 중이며, 특정 용의자의 {MysteryConstants.MEANS}(살해 도구 접근)만을 증명하는 단계입니다. 다음은 미스터리 설명입니다:\n\n{description}\n\n주어진 추론은 '{MysteryConstants.MOTIVE}' 또는 '{MysteryConstants.OPPORTUNITY}(범행 장소 접근)'을 포함하면 안 됩니다. 혹시 아래의 추론이 '{MysteryConstants.MOTIVE}' 또는 '{MysteryConstants.OPPORTUNITY}(범행 장소에 대한 접근)'를 증명하거나 그 증명을 돕는 내용을 담고 있나요?",
                            f"지금은 {MysteryConstants.MEANS}만 증명해야 합니다. '{MysteryConstants.MOTIVE}'나 '{MysteryConstants.OPPORTUNITY}' 관련 내용은 배제되어야 합니다.",
                            conditional='수단이 있다',
                            early_escape_model=model_validator_early_escape_model
                        ),
                        ModelValidator(
                            model_validator_model,
                            f"현재 살인 미스터리를 작성 중이며, 특정 용의자의 {MysteryConstants.OPPORTUNITY}(범행 장소 접근)을 구성하는 단계입니다. 다음은 미스터리 설명입니다:\n\n{description}\n\n주어진 추론은 '{MysteryConstants.MOTIVE}' 또는 '{MysteryConstants.MEANS}(살해 도구 접근)'을 포함하면 안 됩니다. 혹시 아래의 추론이 '{MysteryConstants.MOTIVE}' 또는 '{MysteryConstants.MEANS}(살해 도구 접근)'을 증명하거나 그 증명을 돕는 내용을 담고 있나요?",
                            f"지금은 {MysteryConstants.OPPORTUNITY}만 증명해야 합니다. '{MysteryConstants.MOTIVE}'나 '{MysteryConstants.MEANS}' 관련 내용은 배제되어야 합니다.",
                            conditional='기회가 있다',
                            early_escape_model=model_validator_early_escape_model
                        )
                    ])

            # 템플릿 트리를 생성하고, 모델을 통해 채운다.
            tree = self.complete_structure(
                self.build_structure(
                    depth=depth,
                    bf_factor=bf_factor,
                    chance_to_prune_all=chance_to_prune_all,
                    chance_to_prune=chance_to_prune,
                    root_nodes=root_node
                ),
                model,
                description=description,
                completion_prompt_fn=self.create_completion_prompt(
                    example_completion_trees,
                    example_completion_nodes,
                    example_completion_descriptions,
                    intro=_mm_completion_prompt_intro_,
                    because_clause_after=0,
                    use_complex_facts=False
                ),
                max_retries_on_error=max_retries_on_error,
                inplace=True,
                retry_model=retry_model,
                progress_bar=progress_bar,
                test_prompt=test_completion_prompt,
                validators=validators,
                use_iterative_complete_v2=use_validators
            )
            if not tree.valid:
                return []


            cf_description = ''

            # 의심스러운 사실(레드 헤링) 브랜치를 추가
            if max_num_of_suspicious_facts:
                root_node = [
                    LogicNode(f'{suspect_name}에 대한 의심스러운 사실들', [
                        LogicNode(f'{x} 그리고 이것은 의심스럽다.') for x in suspect_info['red_herrings']
                    ], frozen=True, prunable=False)
                ]

                validators = [StructureValidator()]
                if use_validators:
                    validators.append(ForbiddenTextValidator(
                        forbidden_words=[
                            crime_scene, motive, murder_weapon, '기회', '도구', '동기', '살인 무기', '범죄 현장', '무기'
                        ],
                        reason_why=f"이 단서는 '{MysteryConstants.RED_HERRINGS}'로, 실제 범죄와 직접적 연관성을 갖지 못해야 합니다. 여기서 살인과 직접적으로 연결될 만한 단서는 배제해야 합니다."
                    ))
                    if model_validator_model:
                        validators.append(ModelValidator(
                            model_validator_model,
                            f"현재 살인 미스터리를 작성 중이며, 특정 용의자에 대한 '{MysteryConstants.RED_HERRINGS}(실제로는 범죄와 무관)'를 구성하는 단계입니다. 다음은 미스터리 설명입니다:\n\n{description}\n\n이 추론이 '{MysteryConstants.MOTIVE}', '{MysteryConstants.OPPORTUNITY}', '{MysteryConstants.MEANS}' 세 가지를 모두 증명하거나, 증명을 돕는 내용인가요? 세 가지 중 어느 하나라도 증명할 수 없다면 no라고 답해야 합니다.",
                            "이 사실은 단지 의심스럽게 보일 뿐, 실질적으로 범죄를 증명하지 않아야 합니다.",
                            early_escape_model=model_validator_early_escape_model
                        ))

                cf_description = f'''{suspect_name}{MysteryConstants.determine_josa(suspect_name)} {suspect_info["role"] + MysteryConstants.determine_is(suspect_info["role"])}... 그리고 꽤나 수상한 부분이 있다.'''

                sus_tree = self.complete_structure(
                    self.build_structure(
                        depth=depth,
                        bf_factor=bf_factor,
                        chance_to_prune_all=chance_to_prune_all,
                        chance_to_prune=chance_to_prune,
                        root_nodes=root_node
                    ),
                    model,
                    description=cf_description,
                    completion_prompt_fn=self.create_completion_prompt(
                        sf_example_trees,
                        sf_example_node_completions,
                        sf_example_descriptions,
                        intro=_mm_suspicious_prompt_intro_,
                        because_clause_after=0
                    ),
                    max_retries_on_error=max_retries_on_error,
                    inplace=True,
                    retry_model=retry_model,
                    progress_bar=progress_bar,
                    test_prompt=test_completion_prompt,
                    validators=validators,
                    use_iterative_complete_v2=use_validators
                )
                if not sus_tree.valid:
                    return []
                tree.nodes[0].children.extend(sus_tree.nodes[0].children)

            suspect_trees.append({
                'tree': tree,
                'description': description,
                'cf_description': cf_description,
                'suspect_info': suspect_info,
                'victim_info': victim_info
            })

        return suspect_trees

    def create_chapter_trees(
            self,
            suspect_trees,
            max_num_of_suspicious_facts: int = 1
    ):
        """
        용의자 트리를 기반으로, 무죄 시나리오(innocent_tree)와 유죄 시나리오(murderer_tree)로 나누어 트리를 생성합니다.

        유죄 시나리오에서는 용의자가 수단, 동기, 기회를 모두 가지고 있고, 
        무죄 시나리오에서는 3가지를 모두 만족하지 못하게 됩니다.

        :param suspect_trees: 이미 생성된 용의자 트리 리스트
        :param max_num_of_suspicious_facts: 의심스러운 사실(레드 헤링) 개수
        """
        for sidx, s in enumerate(suspect_trees):
            template = deepcopy(s['tree'])
            t = deepcopy(s['tree'])

            # 의심스러운 사실이 들어있지 않은 브랜치만 남기고 2개만 선택(무죄 시나리오)
            t.nodes[0].children = random.sample([x for x in t.nodes[0].children if '의심' not in x.value.lower()], 2)

            # 무죄 트리에는 (있는 경우) 의심스러운 사실도 추가
            if max_num_of_suspicious_facts:
                t.nodes[0].children.extend([x for x in template.nodes[0].children if '의심' in x.value.lower()])

            suspect_trees[sidx]['innocent_tree'] = t

            # 유죄 트리는 MMO(3가지) 브랜치만 포함 + 일부 의심스러운 사실(원한다면) 추가
            suspect_trees[sidx]['murderer_tree'] = deepcopy(s['tree'])
            suspect_trees[sidx]['murderer_tree'].nodes[0].children = [x for x in template.nodes[0].children if any([y in x.value.lower() for y in ['수단', '동기', '기회']])]

            if max_num_of_suspicious_facts:
                suspect_trees[sidx]['murderer_tree'].nodes[0].children.extend(
                    random.sample([x for x in template.nodes[0].children if '의심' in x.value.lower()], max_num_of_suspicious_facts - 1)
                )

        return suspect_trees

    def create_chapter(
            self,
            model: Model,
            suspect_trees: List[Dict[str, any]],
            facts_only: bool = False,
            validate_model: Model = None,
            validation_tries:int = 5
    ) -> List[Dict[str, any]]:
        """
        스토리가 길어지면 사실(fact)들을 전부 포함시키기 어려우므로, 용의자별로 '장(chapter)'을 따로 작성합니다.

        :param model: 스토리를 생성할 때 사용할 모델
        :param suspect_trees: 'murderer_tree'와 'innocent_tree'를 포함한 용의자 트리 목록
        :param validate_model: 생성된 스토리가 사실을 모두 포함했는지 검증하는 모델(옵션)
        :return: 각 용의자별로 'murderer_chapter'와 'innocent_chapter'가 추가된 결과
        """

        # 용의자별로 장을 생성
        for sidx, s in enumerate(suspect_trees):
            description = s['description']

            # motive 관련 내용을 제거한 innocent_description 생성
            desc = []
            for line in description.split('\n'):
                if '동기' in line.lower():
                    continue
                desc.append(line)

            innocent_description = '\n'.join(desc)

            # 먼저 유죄(진짜 살인자) 시나리오 작성
            assert len(s['murderer_tree'].nodes[0].children) == 3, '트리 구조가 잘못되었습니다.'
            prompt = create_story_prompt__facts_only(description, s['murderer_tree'])

            output, _ = self.inference(prompt, model)

            unsupported = -1

            # 필요하면, 이 장에서 언급된 사실들이 정말 스토리에 반영됐는지 검증
            if validate_model is not None:
                for _ in range(validation_tries):
                    new_output, new_unsupported = self.fact_recall_story_validation(output, s['murderer_tree'], validate_model)

                    if output == new_output:
                        break
                    elif unsupported == -1 or unsupported >= new_unsupported:
                        output = new_output
                        unsupported = new_unsupported
                    else:
                        break

            suspect_trees[sidx]['murderer_chapter'] = output

            # 무죄 시나리오 작성
            assert len(s['innocent_tree'].nodes[0].children) == 3, '트리 구조가 잘못되었습니다.'
            prompt = create_story_prompt__facts_only(innocent_description, s['innocent_tree'])

            output, _ = self.inference(prompt, model)
            unsupported = -1

            if validate_model is not None:
                for _ in range(validation_tries):
                    new_output, new_unsupported = self.fact_recall_story_validation(output, s['innocent_tree'], validate_model)

                    if output == new_output:
                        break
                    elif unsupported == -1 or unsupported >= new_unsupported:
                        output = new_output
                        unsupported = new_unsupported
                    else:
                        break

            suspect_trees[sidx]['innocent_chapter'] = output

        return suspect_trees

    def fact_recall_story_validation(
            self,
            ctx: str,
            tree: LogicTree,
            model: Model
    ):
        """
        생성된 스토리(문맥, ctx)가 트리의 모든 사실들을 실제로 포함하는지(또는 뒷받침하는지) LLM을 통해 확인합니다.

        :param ctx: 사실이 서술되어야 하는 스토리(문맥)
        :param tree: 트리(여기서 추출된 사실들이 스토리에 포함/지지되어야 함)
        :param model: 사실 검증에 사용할 모델
        """

        facts = list(sorted(list(set([x.value for x in tree.get_facts()]))))
        facts_str = "\n".join([f'{fidx} - {x}' for fidx, x in enumerate(facts, start=1)])

        unsupported = []
        curr_cost = model.total_cost
        pbar = tqdm(facts, desc=f'Validating each fact is supported in the story | cost = {0:.2f}', total=len(facts), disable=True)
        prompt = f'''
다음은 한 {STORY}입니다.

{ctx}

다음은 사실들입니다:

{facts_str}

이 사실들은 {STORY}에서 지지되고 있나요? 다음 형식을 사용해 답변하세요.
"사실 - (사실 번호): (당신의 추론), ANSWER: Yes" 혹은 "ANSWER: No"

각 사실별로 한 라인에 작성하세요. 각 라인에 반드시 '사실 -'을 포함하세요. 그렇지 않으면 프로그램이 출력물을 분석할 수 없습니다. 분석한 내용 외에는 아무것도 출력하지 마세요.
        '''.strip()
        output, _ = self.inference(prompt, model, temperature=0.0)

        pbar.set_description(f'Validating each fact is supported in the story | cost = {model.total_cost - curr_cost:.2f}')

        lines = [x for x in output.split('사실 -') if x != '']
        unsupported_reasons = []

        for f, l in zip(facts, lines):
            if 'ANSWER: Yes'.lower() in l.lower():
                continue
            else:
                unsupported.append(f)
                unsupported_reasons.append(l)

        if len(unsupported) == 0:
            return ctx, len(unsupported)

        facts_str = "\n".join([f'- {x}' for x in facts])
        unsupported_str = "\n".join([f'- {x} 이 사실은 다음과 같은 이유로 지지되지 않았습니다: {r}' for x, r in zip(unsupported, unsupported_reasons)])

        new_story_prompt = f'''
우리는 방금 작성한 이야기를 수정하려고 합니다. 
이야기에 반드시 포함되어야 할 사실들이 있는데, 일부 사실이 지지되지 않았습니다.

---
{STORY}:
{ctx}

---

원래 사실 목록:
{facts_str}

지지되지 않은 사실(이 사실들이 {STORY} 속에서 지지되도록 고쳐주세요):
{unsupported_str}

주의할 점:
- 이미 지지되던 다른 사실들을 무효화하지 않고, 
- 지지되지 않은 사실들이 포함(뒷받침)되도록 이야기에 정보를 추가/수정해야 합니다.
- 수정된 {STORY} 외 다른 내용은 출력하지 마세요.

예시: 
만약 "모모는 고양이다. 나는 멋진 것을 좋아한다" 라는 문장이 주어지고,
"고양이는 객관적으로 멋지다."라는 사실이 지지되지 않았다면,
{STORY}를 "고양이들은 의심의 여지 없이 멋지고, 나는 멋진 것들을 좋아한다. 그래서 내가 고양이 모모를 좋아한다." 
이런 식으로 고쳐서 사실을 지지하도록 만들 수 있습니다.

출력:
        '''.strip()

        new_ctx, _ = self.inference(new_story_prompt, model, temperature=0.0)
        return new_ctx, len(unsupported)
