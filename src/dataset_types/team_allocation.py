import sys
from typing import List, Dict, Union, Callable, Tuple
from functools import partial
import random
from copy import deepcopy


from src.madlib.madlib import Madlib
from src.logic_tree.tree import LogicTree, LogicNode, LogicNodeFactType
from src.dataset_builder import DatasetBuilder
from src.model.openai import OpenAIModel
from src.validators import StructureValidator, ForbiddenTextValidator
from src.utils.constants import *

# 이 스크립트는 dataset_builder.py에 정의된 _base_completion_prompt_intro_를 덮어쓰기 위한 목적을 가집니다.
# 특히, object move 트리를 생성할 때 사용됩니다.
__team_allocation_completion_intro__ = f'''
우리는 매니저(당신)가 사람들을 특정 업무에 배정하는 이야기를 만들고 있습니다. 
당신은 각 인물의 과거 경험, 취향, 성격, 사회적 관계 등을 살펴보며 
누가 어떤 업무에 능숙하고, 누가 어떤 업무에 서투른지를 파악해야 합니다.

이 이야기를 만들기 위해, 우리는 이야기의 윤곽을 잡는 트리 구조를 생성했습니다. 
당신의 임무는 이 트리를 채워, 특정 사실들이 어떻게 성립되는지를 보여주는 논리 트리를 완성하는 것입니다.

업무와 스킬 관련 사실을 다룰 때는, 각 인물의 과거 경험과 개인적 이야기를 부각해주세요. 
예를 들어, "영훈이는 테니스를 잘 못한다"라는 사실이라면, 
"영훈이는 어릴 때부터 운동신경이 좋지 않았다"라든지, 
"영훈이는 지금도 스포츠 경기를 피한다"라는 사실을 통해 
"운동신경이 좋지 않고 스포츠를 피하는 사람이라면 테니스를 잘 하지 못할 가능성이 높다"와 같은 추론을 이어갈 수 있습니다.

협업(팀워크)과 관련된 사실을 다룰 때는, 두 사람이 과거에 어떻게 상호 작용했는지, 사회적 요소에 주목해주세요. 
예를 들어, "영훈이와 민수는 함께 일할 때 시너지가 좋다"라는 사실은 
"영훈이와 민수는 가끔 점심을 함께 먹는다"거나 
"둘은 지난주에 같이 작업해서 평이한 시간 안에 일을 마쳤다" 같은 뉘앙스를 통해 
"두 사람이 함께 시간을 보내고 적절하게 일을 마치는 모습을 보였으므로, 함께 잘 일한다고 볼 수 있다"라는 식으로 추론을 전개할 수 있습니다.

무엇보다, 이 사실들은 흥미롭고 인물 중심적이어야 합니다. 너무 건조하거나 일반적인 내용은 피해주세요.

논리 트리는 트리 구조로, 자식 노드의 사실들이 결합되어 부모 노드를 함의하도록 구성됩니다. 
이는 '{NodeTypeConstants.DEDUCED_CONCLUSION}'을 증명하기 위해 필요한 중간 추론 단계를 자연어로 보여주는 구조입니다.

이 단계에서 당신은 '추론 보강' 작업을 수행합니다. 
즉, 하나의 서브트리를 완성하기 위해 자식 노드들의 사실을 채워 넣어야 합니다. 
각 단계를 작성할 때, 주어진 구조(배열된 자식 노드, 사실 분류 등)에 맞추어야 합니다.

'{NodeTypeConstants.FACT_FROM_STORY}'은 이야기에서 명시적으로 언급되는 사실들이며, 
'{NodeTypeConstants.COMMONSENSE}'은 일반적으로 많은 사람이 동의하는 상식 수준의 사실들입니다. 
'{NodeTypeConstants.COMPLEX_FACT}'는 단순 사실들이 결합되어 도출되는 복합적 사실이며, 추후(재귀적 호출) 다른 부분에서 확장될 수 있습니다.

각 사실은 {NodeTypeConstants.DEDUCED_CONCLUSION}을 이끌어내는 데 꼭 필요한 근거여야 합니다. 
서로 중복되거나 불필요한 사실은 지양해주세요.

가능하다면 대명사 대신 인물의 이름을 직접 사용하세요. 
이미 특정 이름을 알고 있다면, "그/그녀" 대신 직접 그 이름을 써주세요.
'''.strip()


class TeamAllocationDataset(DatasetBuilder):
    """
    DatasetBuilder를 상속하여, '팀 배정'과 관련된 기능을 확장한 클래스입니다.

    이 클래스는 사람들을 서로 다른 업무 그룹에 배정하는 상황에서 
    각 인물의 스킬 및 협업 관계를 반영하고, 
    해당 사실들을 논리 트리(LogicTree) 구조로 확장, 검증하는 로직을 포함합니다.
    """

    def build_assignment(
            self,
            people: List[str],
    ):
        """
        이 함수는 각 사람의 스킬 레벨과 협업 점수를 포함하는 행렬을 생성합니다.
        최적 해 조합은 다른 모든 조합보다 높은 점수를 갖도록 보장합니다.

        여기서 사용하는 점수 계산은 다음과 같습니다(의사코드):

        score = Person_1_skill_1 + Person_2_skill_2 + Person_3_skill_2 + Relation_of_Person_2_and_3

        예를 들어, 첫 번째 인물은 스킬 1에 할당되고, 두 번째와 세 번째 인물은 스킬 2에 할당된다고 할 때,
        해당 인물들의 스킬 점수 합과, 두 번째와 세 번째 인물 간의 협업 점수를 더한 값으로 스코어를 결정합니다.

        :param people: 사람 이름 리스트
        :return: people_levels(스킬/협업 매트릭스), best_pair(최적 조합), 모든 조합 리스트
        """

        N = 0
        GOOD = 3
        OKAY = 2
        BAD = 1

        best_pair = [[people[0]], list(sorted([people[1], people[2]]))]

        def asgn(x=0, y=None):
            if x > 2:
                return x
            return random.sample([BAD, OKAY, GOOD][x:y], 1)[0]

        paired_score = asgn()
        people_levels = {
            people[0]: {'skills': [asgn(), BAD], 'cooperation': [N, BAD, BAD]},
            people[1]: {'skills': [BAD, asgn()], 'cooperation': [BAD, N, paired_score]},
            people[2]: {'skills': [BAD, asgn()], 'cooperation': [BAD, paired_score, N]},
        }
        
        def resample_score():
            paired_score = asgn()
            resampled = {
                people[0]: {'skills': [asgn(), BAD], 'cooperation': [N, BAD, BAD]},
                people[1]: {'skills': [BAD, asgn()], 'cooperation': [BAD, N, paired_score]},
                people[2]: {'skills': [BAD, asgn()], 'cooperation': [BAD, paired_score, N]},
            }
            return resampled



        

        def score(pair):
            pair_sum = 0
            pair_sum += people_levels[pair[0][0]]['skills'][0]
            pair_sum += sum([people_levels[pair[1][0]]['skills'][1], people_levels[pair[1][1]]['skills'][1]])
            pair_sum += people_levels[pair[1][0]]['cooperation'][people.index(pair[1][1])]

            return pair_sum

        def score_pairs(pairs):
            gold_pair = None
            scored_pairs = []
            for p in pairs:
                s = score(p)
                if p == best_pair:
                    gold_pair = [s, p]
                    continue
                scored_pairs.append([s, p])

            return [gold_pair, *scored_pairs]

        def gen_pairs():
            pairs = []
            for i in people:
                for j in people:
                    for x in people:
                        if len(list(set([i, j, x]))) != 3:
                            continue
                        p = [[i], list(sorted([j, x]))]
                        if p in pairs:
                            continue
                        pairs.append(p)
            return pairs
        
        # 수정: 원래의 코드에서는 delta <= 0인 경우를 체크하지 않았으나, 수정된 코드에서는 delta > 0인 경우에만 후속 작업을 수행.
        while True:
            scored_pairs = score_pairs(gen_pairs())
            delta = scored_pairs[0][0] - max([x[0] for x in scored_pairs[1:]])
            if 0 < delta < 2:
                break
            people_levels = resample_score()

        assert delta > 0, 'WRONG!'

        # gold assignment와 비교했을 때 다른 조합의 점수를 적절히 조정하여
        # gold assignment만이 최대 점수를 갖도록 유지하는 과정입니다.

        # 위의 random 재추출 과정이 있기 때문에 아래 코드는 더이상 필요하지 않으며 오히려 오류를 만들 우려가 있음.
        # while delta > 2 and update_idx < max_updates:
        #     update_idx += 1

        #     second_best = list(sorted(scored_pairs[1:], key=lambda x: x[0]))[0][1]

        #     group = 1 if random.random() > 0.75 else 0
        #     if group == 0:
        #         # 두 번째로 좋은 조합에 속한 첫 번째 인물의 스킬을 조금 수정
        #         people_levels[second_best[0][0]]['skills'][0] = asgn(x=people_levels[second_best[0][0]]['skills'][0])
        #     else:
        #         inc = 1 if random.random() > 0.66 else 0
        #         if inc == 0:
        #             person = random.sample(second_best[1], 1)[0]
        #             people_levels[person]['skills'][1] = asgn(x=people_levels[person]['skills'][1])
        #         else:
        #             people_levels[second_best[1][0]]['cooperation'][people.index(second_best[0][0])] = asgn(
        #                 x=people_levels[second_best[1][0]]['cooperation'][people.index(second_best[0][0])])
        #             people_levels[second_best[0][0]]['cooperation'][people.index(second_best[1][0])] = \
        #                 people_levels[second_best[0][0]]['cooperation'][people.index(second_best[1][0])]

        #     scored_pairs = score_pairs(gen_pairs())
        #     delta = scored_pairs[0][0] - max([x[0] for x in scored_pairs[1:]])

        #     assert delta > 0, 'WRONG!'
        return people_levels, best_pair, [x[1] for x in scored_pairs]

    def create_facts(self, people_levels, people, skills):
        """
        build_assignment에서 만든 스킬/협업 점수를 바탕으로 
        사실(F)의 집합을 생성합니다.

        :param people_levels: build_assignment의 결과로 나온 스킬/협업 정보
        :param people: 사람 이름 리스트
        :param skills: 스킬(2개)
        :return: 사실(F) 리스트
        """

        prof_levels = {1: '못 한다', 2: '할 수 있다', 3: '잘 한다'}
        coop_levels = {1: '못 일한다', 2: '일할 수 있다', 3: '잘 일한다'}

        facts = [
            f'{TeamAllocationConstants.add_josa(name, "subject")} {TeamAllocationConstants.add_josa(skill, "object")} {prof_levels[prof]}.'
            for name, vals in people_levels.items()
            for skill, prof in zip(skills, vals['skills'])

        ]

        facts.append(f'{TeamAllocationConstants.add_josa(people[0], "conjunction")} {TeamAllocationConstants.add_josa(people[1], "subject")} 같이 {coop_levels[people_levels[people[0]]["cooperation"][1]]}.')
        facts.append(f'{TeamAllocationConstants.add_josa(people[0], "conjunction")} {TeamAllocationConstants.add_josa(people[2], "subject")} 같이 {coop_levels[people_levels[people[0]]["cooperation"][2]]}.')
        facts.append(f'{TeamAllocationConstants.add_josa(people[1], "conjunction")} {TeamAllocationConstants.add_josa(people[2], "subject")} 같이 {coop_levels[people_levels[people[1]]["cooperation"][2]]}.')
        return facts

    def create_fact_trees(
            self,
            model: OpenAIModel,
            facts: List[str],
            tasks,
            description: str,
            example_completion_trees: List[LogicTree],
            example_completion_nodes: List[LogicNode],
            example_completion_descriptions: List[str],
            depth: int = 4,
            bf_factor: Dict[int, float] = None,
            chance_to_prune_all: float = 0.45,
            chance_to_prune: float = 0.5,
            max_retries_on_error: int = 1,
            progress_bar: bool = False,
            test_complete_structure_prompt: bool = False,
            retry_model: OpenAIModel = None,
            use_complex_facts: bool = True,
            use_validators: bool = False
    ):
        """
        facts(사실 리스트)와 시나리오(description)를 사용해 논리 트리(LogicTree)를 생성하고,
        모델을 통해 노드를 채워 완성된 트리 구조를 얻습니다.

        :param model: DatasetBuilder가 사용하는 모델(OpenAIModel 등)
        :param facts: create_facts 함수에서 만든 사실 리스트
        :param tasks: 업무(스킬) 리스트
        :param description: 시나리오 설명
        :param example_completion_trees: ICL 예시로 사용할 LogicTree 목록
        :param example_completion_nodes: ICL 예시로 사용할 LogicNode 목록
        :param example_completion_descriptions: ICL 예시에서 사용하는 시나리오 설명
        :param depth: 트리의 최대 깊이
        :param bf_factor: 각 레벨에서의 분기 계수(깊이에 따른 노드 분기 조절)
        :param chance_to_prune_all: 전체를 prune할 확률
        :param chance_to_prune: 각 노드를 prune할 확률
        :param max_retries_on_error: 에러 발생 시 재시도 횟수
        :param progress_bar: 진행 상황을 표시할지 여부
        :param test_complete_structure_prompt: 구조 완성 프롬프트를 테스트할지 여부
        :param retry_model: 재시도 시 사용할 모델
        :param use_complex_facts: Complex Facts(복합 사실)을 사용할지 여부
        :param use_validators: 유효성 검사(Validators)를 사용할지 여부
        :return: 완성된 LogicTree 객체
        """

        nodes = [LogicNode(f'{x} ') for x in facts]

        validators = [
            StructureValidator(),
            ForbiddenTextValidator(
                forbidden_words=[
                    *tasks
                ]
            )
        ]

        tree = self.complete_structure(
            self.build_structure(
                depth=depth,
                bf_factor=bf_factor,
                chance_to_prune_all=chance_to_prune_all,
                chance_to_prune=chance_to_prune,
                root_nodes=[LogicNode(description, nodes, frozen=True, prunable=False)]
            ),
            model,
            description=description,
            completion_prompt_fn=self.create_completion_prompt(
                example_completion_trees,
                example_completion_nodes,
                example_completion_descriptions,
                intro=__team_allocation_completion_intro__,
                because_clause_after=-1,
                use_complex_facts=use_complex_facts
            ),
            max_retries_on_error=max_retries_on_error,
            inplace=True,
            progress_bar=progress_bar,
            retry_model=retry_model,
            test_prompt=test_complete_structure_prompt,
            use_iterative_complete_v2=use_validators,
            validators=validators
        )

        return tree
