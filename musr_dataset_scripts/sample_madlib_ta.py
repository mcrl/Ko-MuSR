import json
import copy
import sys
import time
from copy import deepcopy
from pathlib import Path
import random
import pprint
import re
import traceback

from functools import partial
from src import cache,set_seed
from src.model import OpenAIModel
from src.logic_tree.tree import LogicTree, LogicNode, LogicNodeFactType
from src.madlib.madlib import Madlib
from src.utils.paths import OUTPUT_FOLDER
from src.utils.constants import MysteryConstants, TeamAllocationConstants
from src.dataset_types.team_allocation import TeamAllocationDataset

set_seed(500)
def respect_article(item, people):
    if any([item.startswith(name) for name in people]):
        return item
    return f'the {item}'

def respect_plural(item):
    if item.endswith('s'):
        return f'{item} are'
    return f'{item} is'

cLogicNode = partial(LogicNode, fact_type=LogicNodeFactType.COMMONSENSE)

"""추론을 만들기 위한 ICL 예시입니다. 자세한 내용은 datasetbuilder를 참조하세요."""

example1_description = '민수와 예림이는 노래방에 있다.'
example1_tree = LogicTree(
    nodes=[
        LogicNode(example1_description, [
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
example1_node_completion_tree =  LogicNode(example1_description, [
    LogicNode('예림이는 맥주를 쓰레기통으로 옮긴다.', [
        LogicNode('민수는 맥주가 쓰레기통으로 옮겨지는 것을 보지 못했다.', [
            LogicNode('예림이는 민수를 속여서 "저쪽"을 보게 했다.'),
            LogicNode('예림이는 민수에게 쓰레기통의 반대 방향을 가리켰다.'),
            cLogicNode('누군가를 속여서 다른 곳을 보게 만들면, 반대쪽에서 일어나는 일을 볼 수 없다.')
        ])
    ])
])


example_descriptions = [example1_description]
example_trees = [example1_tree]
example_node_completions = [example1_node_completion_tree.children[0].children[0]]

def remove_prepended_numbers(strings):
    return [re.sub(r'^\d+\.\s*', '', s) for s in strings]

def main():
    # CACHE
    # cache.enable()

    # PARAMS (if not with a comment, look at the Object Placements dataset class for more info.)

    max_examples = 1

    gpt35 = OpenAIModel(engine='gpt-3.5-turbo', api_endpoint='chat', api_max_attempts=30, temperature=1.0, max_tokens=1500, num_samples=1, prompt_cost=0.0015/1000, completion_cost=0.002/1000)
    gpt16k35 = OpenAIModel(engine='gpt-3.5-turbo-16k', api_endpoint='chat', api_max_attempts=30, temperature=1.0, max_tokens=2400, num_samples=1, prompt_cost=0.003/1000, completion_cost=0.004/1000)
    gpt4 = OpenAIModel(engine='gpt-4', api_max_attempts=30, api_endpoint='chat', temperature=1.0, top_p=1.0, max_tokens=2400, num_samples=1, prompt_cost=0.03/1000, completion_cost=0.06/1000)
    gpt4omini = OpenAIModel(engine='gpt-4o-mini', api_max_attempts=30, api_endpoint='chat', temperature=1.0, top_p=1.0, max_tokens=2400, num_samples=1, prompt_cost=0.15/1000000, completion_cost=0.6/1000000)
    gpt4o = OpenAIModel(engine='gpt-4o', api_max_attempts=30, api_endpoint='chat', temperature=1.0, top_p=1.0, max_tokens=2400, num_samples=1, prompt_cost=2.5/1000000, completion_cost=10/1000000)

    model_to_use = gpt4o
    
    raw_idx = 0


    raw_idx += 1
    creator = TeamAllocationDataset()

    # Sample a scenario to build the story around.
    
    madlib = creator.build_madlib(
        model_to_use,
        things_to_create_names=['scenario_descriptions'],
        things_to_create_description=[
            "사람들을 서로 다른 업무 그룹에 배치해야 하는 상황에 대한 시나리오를 작성하세요. 각 업무를 해결하기 위해 필요한 기술이 서로 다르고 고유해야 합니다. 시나리오만 작성하고, 번호는 매기지 마세요. 언급할 업무는 2개여야 합니다."
        ],
        examples_of_things_to_create=[
            [
                "당신은 새로운 카페를 열 계획입니다. 메뉴 개발과 인테리어 디자인을 위해 카페 팀을 어떻게 효과적으로 구성해야 할까요?",
                "한 기업이 새로운 소프트웨어를 출시하려고 합니다. 마케팅 전략과 기술 지원 팀을 어떻게 배분하여 성공적인 론칭을 이끌어낼 수 있을까요?",
                "당신은 지역 커뮤니티 센터에서 청소년 프로그램을 운영하고 있습니다. 프로그램 기획과 안전 관리 팀을 어떻게 나누어야 할까요?",
                "대규모 행사인 음악 축제를 준비하고 있습니다. 무대 운영과 관객 안전 관리 팀을 어떻게 구성해야 할까요?",
                "당신은 새로운 패션 브랜드를 런칭하려고 합니다. 디자인 팀과 마케팅 팀을 어떻게 각각의 전문성을 살려 배치해야 할까요?"
            ]
        ]
        ,max_n_creations=1000
    )

    out_file = "domain_seed/ta_scenario_descriptions.json"
    out_file = Path(out_file)
    madlib.save({'scenario_descriptions': out_file})


if __name__ == "__main__":
    main()