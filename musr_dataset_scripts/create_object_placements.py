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

from src import set_seed

from src import cache
from src.model import OpenAIModel
from src.logic_tree.tree import LogicTree, LogicNode, LogicNodeFactType
from src.utils.paths import OUTPUT_FOLDER, ROOT_FOLDER
from src.utils.constants import ObjectPlacementConstants, Conjunctions
from src.dataset_types.object_placements_dataset import ObjectPlacementsDataset
from functools import partial
from src.utils.constants import ObjectPlacementConstants, STORY
from src.madlib.madlib import Madlib
from argparse import ArgumentParser


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
    parser = ArgumentParser()
    parser.add_argument('--cache', action='store_true', help='Enable caching')
    parser.add_argument("--model", type=str, default='gpt-4o-mini', help="Model to use", choices=["gpt-4o-mini", "o1", "o3-mini", "gpt-4o", "gpt-4"])
    parser.add_argument("--seed", type=int, default=500, help="Random seed")
    parser.add_argument("--num_samples", type=int, default=1, help="Maximum number of examples to generate")
    args = parser.parse_args()
    
    
    max_sequence_len = 3
    chance_to_see = 0.33

    tree_depth = 3
    max_structure_completion_retries = 3

    verbose = False
    use_validators = True

    # CREATION LOGIC
    total_cost = 0

    one_million= 1000000
    gpt35 = OpenAIModel(engine='gpt-3.5-turbo', api_endpoint='chat', api_max_attempts=30, temperature=1.0, max_tokens=1500, num_samples=1, prompt_cost=0.0015/1000, completion_cost=0.002/1000)
    gpt16k35 = OpenAIModel(engine='gpt-3.5-turbo-16k', api_endpoint='chat', api_max_attempts=30, temperature=1.0, max_tokens=2400, num_samples=1, prompt_cost=0.003/1000, completion_cost=0.004/1000)
    gpt4 = OpenAIModel(engine='gpt-4', api_max_attempts=30, api_endpoint='chat', temperature=1.0, top_p=1.0, max_tokens=2400, num_samples=1, prompt_cost=0.03/1000, completion_cost=0.06/1000)
    gpt4omini = OpenAIModel(engine='gpt-4o-mini', api_max_attempts=30, api_endpoint='chat', temperature=1.0, top_p=1.0, max_tokens=2400, num_samples=1, prompt_cost=0.15/one_million, completion_cost=0.6/one_million)
    gpt4o = OpenAIModel(engine='gpt-4o', api_max_attempts=30, api_endpoint='chat', temperature=1.0, top_p=1.0, max_tokens=2400, num_samples=1, prompt_cost=2.5/one_million, completion_cost=10/one_million)
    o1 = OpenAIModel(engine='o1', api_max_attempts=30, api_endpoint='chat', temperature=1.0, top_p=1.0, max_tokens=2400, num_samples=1, prompt_cost=15/one_million, completion_cost=60/one_million)
    o3_mini = OpenAIModel(engine='o3-mini', api_max_attempts=30, api_endpoint='chat', temperature=1.0, top_p=1.0, max_tokens=2400, num_samples=1, prompt_cost=1.1/one_million, completion_cost=4.4/one_million)

    set_seed(args.seed)

    if args.cache:
        cache.enable()

    if args.model == 'gpt-4o-mini':
        all_models= gpt4omini
    elif args.model == 'o1':
        all_models = o1
    elif args.model == 'o3-mini':
        all_models = o3_mini
    elif args.model == 'gpt-4o':
        all_models = gpt4o
    elif args.model == 'gpt-4':
        all_models = gpt4

    model_to_use = all_models
    validator_model = all_models
    validator_model_early_escape = gpt4o

    max_examples = args.num_samples
    out_file = OUTPUT_FOLDER / f'custom_object_placements_{args.model}_{args.seed}.json'

    if out_file:
        out_file.parent.mkdir(exist_ok=True, parents=True)

    max_examples = args.num_samples

    dataset = []

    previous_samples = []

    __idx = 0
    raw_idx = 0
    max_idx = int(max_examples * 10)

    madlib_path = ROOT_FOLDER / 'domain_seed' / 'op_scenario_descriptions.json'
    madlib = Madlib({"scenario_descriptions": madlib_path})

    while __idx < max_examples and raw_idx < max_idx:
        print(f"STORY: {__idx+1}")

        raw_idx += 1
        creator = ObjectPlacementsDataset()

        # This scenario acts like a high level description of the story.
        descriptions, _, previous_samples = creator.sample_madlib(madlib, ['scenario_descriptions'], '{scenario_descriptions}', previously_sampled=previous_samples)
        print(f"DESCRIPTIONS: {descriptions}")
        description = descriptions[0]
        if not description:
            exit(1)

        # Here we prompt a model to give us details about the scenario. For example, the items that will be moved, the
        # people who will move them, why they are moving things around etc.  We found this extremely helpful in creating
        # cohesive stories about people moving things that don't sound too "dream-like".
        # We also ask the model to produce the "moves" (who moved what to where) in one go.
        # This forces the model to create a "narrative" about why things are happening before we make the actual story.
        prompt = f'''
당신은 주어진 시나리오의 짧은 설명을 바탕으로 극적인 이야기의 개요를 만들 것입니다. 이를 위해 세 명의 인물을 만들고, 각각의 역할과 동기를 설정할 것입니다.

그다음, 이야기가 전개되는 장면을 설정할 것입니다. 세 명의 인물이 함께 무엇을 하고 있는지(이야기의 목표)를 결정합니다.

이후, 세 개의 "이동" 목록을 만듭니다. "이동"이란 한 인물이 특정 물건을 한 장소에서 다른 장소로 옮기는 행동을 의미합니다. 이동되는 물건은 작고 만질 수 있는 것이어야 하며, 위치는 해당 물건을 담을 수 있는 공간이어야 합니다. 예를 들어, 선반은 하나의 위치가 될 수 있으며, 커피 한 봉지는 하나의 물건이 될 수 있습니다. 이는 선반이 합리적으로 커피 봉지를 보관할 수 있기 때문입니다.

각 이동에 대해, 왜 이 행동이 이루어졌는지 그리고 이야기와 어떤 관련이 있는지를 설명할 것입니다. 가장 중요한 점은, 행동의 정당성이 다른 등장인물에 의존해서는 안 된다는 것입니다. 우리는 독자가 이야기와 인물의 관찰을 기반으로 질문을 만들 수 있도록 각 인물이 자신의 활동을 수행하고 물건을 옮기는 방식으로 구성해야 합니다.

규칙:
1) 인물을 묘사할 때 실제 이름을 사용하고, 역할이 주어진 설명에 맞아야 합니다.
2) 이야기의 동기와 장소는 인물과 그들의 역할에 적절해야 합니다.
3) 이야기 개요에는 세 명의 인물이 비슷한 목표를 향해 협력하며, 하나의 주요 사건이 포함되어야 합니다. 예를 들어, 고객을 위한 커피를 만드는 상황이 될 수 있습니다.
4) 이동들은 지금까지의 이야기 흐름, 인물의 역할, 장소와 일관성이 있어야 합니다. 예를 들어, 카페에서 고객이 우유를 옮기는 것은 적절하지 않습니다.
5) 이동하는 물건은 작고 쉽게 이동할 수 있는 유형의 것이어야 합니다. "공연"과 같은 추상적인 개념이나 "텔레비전"과 같은 대형 물건은 사용하지 마세요. 대신 "아이폰", "노트북", "수첩"처럼 작고 손쉽게 옮길 수 있는 물건을 사용하세요.
6) 선택하는 장소는 물건을 보관할 수 있어야 합니다. 예를 들어, "골프채"가 물건이라면, 모든 장소는 골프채를 보관할 수 있는 공간이어야 합니다. "코트 주머니" 같은 부적절한 장소를 설정하지 마세요.
7) 물건을 이동시키는 이유는 해당 물건을 옮기는 인물의 입장에서 설명해야 하며, 다른 인물과 관련지어서는 안 됩니다. 이야기와 장소에 대한 세부 사항은 포함할 수 있지만, 다른 사람을 포함해서는 안 됩니다. (이후 독자가 이야기 속 질문과 답변을 쉽게 만들 수 있도록 하기 위함입니다.)
8) 이야기를 만들 때는 두 개의 서로 다른 물건, 세 명의 인물, 네 개의 장소만 활용하세요. 한 사람이 한 물건을 두 번까지 이동시킬 수 있지만, 세 번 이상은 안 됩니다.
9) 장소의 주인을 지칭하지 마세요. 예를 들어, "그의 책상에서 그녀의 선반으로"나, "철수의 책상에서 영희의 선반으로"라고 대명사나 특정 인물의 이름을 넣지 마세요. 이렇게 하면 결과물을 분석하기 어렵게 만듭니다.
10) 이동의 형식은 반드시 다음과 같이 작성해야 합니다. 그래야 파이썬 프로그램이 이를 분석할 수 있습니다.
    "[이름]이/가 [물건]을(를) [출발 위치]에서 [도착 위치](으)로 옮긴다."
    이 형식을 절대 벗어나지 마세요.
11) 이동 이유를 설명할 때, 다른 물건의 이름을 포함해서는 안 됩니다. 예를 들어, 이동하는 물건이 "카드"와 "사과"라면, "카드"의 이동 이유를 설명할 때 "사과"를 언급해서는 안 됩니다.
12) 세 개의 이동은 서로 다른 두 개의 물건을 활용해야 합니다.
13) 동일한 물건이 두 번 이상 등장할 경우, 이전 이동을 반영하여 위치가 일관되게 유지되어야 합니다.

예제:

이야기 개요: 영희는 그녀의 직장에서 커피를 만들고 있으며, 고객을 위해 아몬드 우유를 사용하려 합니다.

출력:

인물 1
{ObjectPlacementConstants.NAME}: 영희
{ObjectPlacementConstants.ROLE}: 바리스타
{ObjectPlacementConstants.MOTIVE}: 영희는 민수에게 아몬드 우유가 들어간 커피를 만들어 주고 싶어 합니다.

인물 2
{ObjectPlacementConstants.NAME}: 민수
{ObjectPlacementConstants.ROLE}: 고객
{ObjectPlacementConstants.MOTIVE}: 민수는 마감 기한이 다가오고 연구 실험이 실패하는 등 힘든 한 주를 보내고 있습니다. 그래서 그가 가장 좋아하는 아몬드 우유 커피를 마시고 싶어 합니다.

인물 3
{ObjectPlacementConstants.NAME}: 철수
{ObjectPlacementConstants.ROLE}: 카페 직원
{ObjectPlacementConstants.MOTIVE}: 철수는 카페에서 일하는 첫날입니다. 그는 고객을 위해 모든 테이블을 깨끗하게 유지하기 위해 열심히 일하고 있습니다.

이야기 개요:
민수는 마감 기한이 다가오고 실험이 실패하는 등 힘든 한 주를 보내고 있어, 그가 가장 좋아하는 아몬드 우유 커피가 절실히 필요했습니다. 영희는 숙련된 바리스타로 고객들에게 따뜻한 음료수를 만드는 것을 좋아합니다. 한편, 철수는 신입 직원이라 계속해서 어수선하게 실수하고 있었습니다. 영희는 그를 바쁘게 만들기 위해 청소 업무를 맡겼습니다.

이동:

이동 1 - 철수가 아몬드 우유를 냉장고에서 뒷선반으로 옮긴다.
{ObjectPlacementConstants.MOVER}: 철수
{ObjectPlacementConstants.OBJECT}: 아몬드 우유
{ObjectPlacementConstants.SRC}: 냉장고
{ObjectPlacementConstants.DEST}: 뒷선반
{ObjectPlacementConstants.REASON} - 민수는 모든 사람이 더 효율적으로 일할 수 있도록 냉장고를 정리하고 있습니다.

이동 2 - 영희가 커피 봉지를 뒷선반에서 앞 카운터로 옮긴다.
{ObjectPlacementConstants.MOVER}: 영희
{ObjectPlacementConstants.OBJECT}: 커피 봉지
{ObjectPlacementConstants.SRC}: 뒷선반
{ObjectPlacementConstants.DEST}: 앞 카운터
{ObjectPlacementConstants.REASON} - 영희는 재료가 부족해져서 예비 원두를 가져와야 했습니다.

이동 3 - 영희가 아몬드 우유를 뒷선반에서 냉장고로 옮긴다.
{ObjectPlacementConstants.MOVER}: 영희
{ObjectPlacementConstants.OBJECT}: 아몬드 우유
{ObjectPlacementConstants.SRC}: 뒷선반
{ObjectPlacementConstants.DEST}: 냉장고
{ObjectPlacementConstants.REASON} - 영희는 재료가 너무 오래 방치되었음을 깨닫고, 상하기 전에 다시 냉장고에 넣었습니다.

이제 당신의 차례입니다! 두 가지 조건을 특히 조심하세요!
11) 이동 이유를 설명할 때, 다른 물건의 이름을 포함해서는 안 됩니다. 예를 들어, 이동하는 물건이 "카드"와 "사과"라면, "카드"의 이동 이유를 설명할 때 "사과"를 언급해서는 안 됩니다.
12) 세 개의 이동은 서로 다른 두 개의 물건을 활용해야 합니다.

이야기 개요: {description}

출력:
'''.strip()

        if verbose:
            print('--- MADLIB PROMPT ---')
            print(prompt)

        output, _ = creator.inference(prompt, validator_model_early_escape)

        if verbose:
            print("=== MADLIB OUTPUT ===")
            print(output)

        # Here we will parse out all the info we want from the generation
        items = []
        people = []
        people_data = []
        moves = []
        move_strs = []
        locations = []
        world_state = []

        try:
            lines = output.split('\n\n')

            for c in lines[0:3]:
                info = c.split('\n')
                people_data.append({
                    'name': info[1].replace(f'이름: ', '').strip(),
                    'role': info[2].replace(f'{ObjectPlacementConstants.ROLE}: ', '').strip(),
                    'motivation': info[3].replace(f'{ObjectPlacementConstants.MOTIVE}: ', '').strip()
                })
                people.append(info[1].replace(f'이름: ', ''))

            story_desc = lines[3].replace('이야기 개요:\n','')

            for move_info in lines[5:]:
                m = move_info.split('\n')
                # print(move_info)
                # print()
                mover_string = f"{ObjectPlacementConstants.MOVER}: "
                move_data = {
                    'mover': m[1].replace(f'{ObjectPlacementConstants.MOVER}: ', '').strip(),
                    'item': m[2].replace(f'{ObjectPlacementConstants.OBJECT}: ', '').strip(),
                    'from': m[3].replace(f'{ObjectPlacementConstants.SRC}: ', '').replace('그의', f'{m[1].replace(mover_string, "")}의').replace('그녀의', f'{m[1].replace(mover_string, "")}의').strip(),
                    'to': m[4].replace(f'{ObjectPlacementConstants.DEST}: ', '').replace('그의', f'{m[1].replace(mover_string, "")}의').replace('그녀의', f'{m[1].replace(mover_string, "")}의').strip(),
                    'justification': m[5].replace(f'{ObjectPlacementConstants.REASON} - ', '').strip()
                }

                moves.append(move_data)

                locations.extend([move_data['from'], move_data['to']])
                items.append(move_data['item'])

                if move_data['item'] not in [x[0] for x in world_state]:
                    world_state.append([move_data['item'], move_data['from']])
                move_str = ObjectPlacementConstants.move(move_data['mover'], move_data['item'], move_data['to'])
                move_strs.append(move_str)

            people = list(sorted(set(people)))
            items = list(sorted(set(items)))

            locations = list(sorted(set(locations)))
            # 사람 이름 제거 (예: '민철의 책상' → '책상')
            locations = [re.sub(r'^[^의]+의\s*', '', loc) for loc in locations]


            items_str = '\n'.join([f'- {x}' for x in items])

            # Gotta have 2 items always.
            if len(items) != 2:
                print("ERROR: WRONG ITEM COUNT")
                continue

            # 장소가 4군데만 나오도록 체크
            if len(locations) != 4:
                print("ERROR: WRONG LOCATION COUNT")
                continue

            # You cannot mention the other object in the justification.
            # yet you can mention the item in the justification.
            
            passed = True
            for m in moves:
                if not passed:
                    break
                the_other_item = items[0] if m['item'] == items[1] else items[1]
                if the_other_item in m['justification']:
                    print("ERROR: YOU CAN'T MENTION AN ITEM IN THE JUSTIFICATION")
                    passed = False
            if not passed:
                continue

            # To check if the moves produced by the model are valid, we recreate them using our code.  This also gives
            # us the belief states and observations of other people in the story.  This could be cleaner, but deadlines.
            max_retries = 100_000
            mri = 0
            sampled = False
            while mri < max_retries:
                mri += 1
                events, beliefs, actual_locs, event_structure = creator.create_sequence_v2(
                    items, deepcopy(locations), people, max_sequence_length=max_sequence_len, chance_subject_sees=chance_to_see, initial_starting_positions=world_state
                )

                retry = False
                assert len(events) == len(move_strs) + 1
                for gidx, (gm, e) in enumerate(zip(move_strs, events[1:])):
                    move = e[0]
                    if move.lower() != gm.lower():
                        retry = True
                        break
                if retry:
                    continue
                sampled = True
                break
            if not sampled:
                raise Exception("Couldn't get a sample from the code that matched the gold move from GPT.")

            question, answers = creator.generate_end_questions(
                ending_beliefs=beliefs[-1], people=people, items=items, locations=locations, event_structure=deepcopy(event_structure)
            )
        except Exception as e:
            # If for any reason something fails, we will just retry the whole loop.
            print("ERROR")
            print(e)
            traceback.print_exc()
            print(flush=True)
            time.sleep(1)
            continue

        print(f"STORY: {__idx}")

        completion_description = f'''
{story_desc}

인물 1:
{ObjectPlacementConstants.NAME}: {people_data[0]['name']}
{ObjectPlacementConstants.ROLE}: {people_data[0]["role"]}
{ObjectPlacementConstants.MOTIVE}: {people_data[0]["motivation"]}

인물 2:
{ObjectPlacementConstants.NAME}: {people_data[1]['name']}
{ObjectPlacementConstants.ROLE}: {people_data[1]["role"]}
{ObjectPlacementConstants.MOTIVE}: {people_data[1]["motivation"]}

인물 3:
{ObjectPlacementConstants.NAME}: {people_data[2]['name']}
{ObjectPlacementConstants.ROLE}: {people_data[2]["role"]}
{ObjectPlacementConstants.MOTIVE}: {people_data[2]["motivation"]}

추론을 만들 때에는 인물들의 역할을 존중하고, 그들이 이야기에 따라 
어떤 사건을 보거나 보지 못하도록 하는 동기를 부여해 주세요. 또한 이러한 동기나 이야기는 여러 "이동"에 걸쳐 일관되게 이어지도록 노력해 주세요.
누군가가 어떤 것을 보았거나 보지 못했다고 명시적으로 말하지 마세요. 당신의 역할은 이를 암시하는 사실들을 제공하는 것입니다. 예를 들어, 지수가 철수가 무언가를 옮기는 것을 보았다고 암시하고 싶다면, "지수는 집안일을 하면서 철수를 보고 있었다" 그리고 "누군가를 보고 있다면, 그 사람이 무엇을 하고 있는지를 보는 것이다"라고 말할 수 있습니다. 누군가가 본 사실을 직접적으로 말하지 않고, 보는 행위의 물리적 조건을 설명하는 방식입니다.
물건이 움직였다는 사실을 언급하거나 이미 등장한 물건을 재사용하지 마세요. 예를 들어, "철수가 자신의 아이폰을 보지 못했다"라는 사실이 있고, 당신이 "유림이는 사과가 움직이는 것을 보지 못했다"를 증명하고 있다면, 철수의 아이폰을 다시 사용하지 마세요. 우리 프로그램은 아이템이 어디에 있는지를 엄격하게 제어하고 있으며, 우리가 고려하지 않은 아이템 배치를 도입하면 안 됩니다.
'''


        for eidx, e in enumerate(event_structure[1:]):
            event_structure[eidx+1]['event'] += f' {Conjunctions.BECAUSE}, {moves[eidx]["justification"]}'

        tree = creator.create_event_trees(
            model_to_use,
            event_structure,
            items=items,
            locations=locations,
            completion_description=completion_description,
            description=description,
            example_completion_trees=example_trees,
            example_completion_nodes=example_node_completions,
            example_completion_descriptions=example_descriptions,
            depth=tree_depth,
            bf_factor={2: 1.0},
            chance_to_prune=0.0,
            chance_to_prune_all=0.0,
            max_retries_on_error=max_structure_completion_retries,
            progress_bar=True,
            test_complete_structure_prompt=False,
            retry_model=model_to_use,
            use_validators=use_validators,
            # model_validator_model=validator_model,
            # model_validator_model_early_escape=validator_model_early_escape
        )

        if verbose:
            print('=== TREE OUTPUT ===')
            print(tree.print_for_gpt(pad_char='> ', pad_space=1))


        # We will now create the story in many chapters.
        # Although these are created in chapters, we take advantage of the autoregressive nature of LLMs.  Specifically,
        # every chapter should start with the story so far, and then ask the model to continue it.
        failed_parse = False
        chapters = []
        story_so_far = ''
        for loop_idx, __n in enumerate(tree.nodes[0].children):
            n = deepcopy(__n)
            if n.value == '이야기 시작':
                # The opening scene "chapter" is meant to introduce the setting but also say that everyone knows where
                # everything is initially (a starting point.)
                facts_str = '\n'.join([f'- {respect_article(respect_plural(x), people)} at {respect_article(y, people)}' for x, y in actual_locs[0].items()])
                facts_str = '\n'.join([f'- {ObjectPlacementConstants.add_josa(x, "subject")} {y}에 있다.' for x, y in actual_locs[0].items()])

                opening_prompt = f"""
우리는 사람들이 각자의 동기에 따라 물건을 옮기고, 물건의 위치를 추리하는 이야기를 만들고 있습니다.
당신은 짧은 오프닝 장면 묘사를 만들어야 합니다.
우리는 물건과 그 위치를 제공할 것이며, 오직 해당 리스트에 언급된 아이템과 위치에 대해서만 이야기해야 합니다. 
반드시 리스트에 있는 각 물건과 해당 위치를 이야기 속에 포함시켜 주세요. 또한 모든 등장인물이 이 물건들의 위치를 알고 있었다는 점을 반드시 언급해야 합니다!

주어진 설명을 바탕으로 장면을 유추할 수 있지만, 리스트에 제공된 사실로만 이야기를 구성해야 합니다.

모든 사람이 모든 물건의 위치를 알고 있다는 사실을 반드시 밝혀 주세요. 예를 들어 "그들은 모두 각 물건이 어디에 있는지 알고 있었다" 혹은 이와 유사한 문구로 표현할 수 있습니다. 이 조건을 {STORY}와 일관성 있게 연결하도록 해보세요. 예를 들어, 누군가가 정확한 위치를 모르더라도 "모두 그 물건이 해당 장소 어딘가에 있다는 사실은 알고 있었고, 다른 물건이 어디에 있다는 것 또한 알고 있었다"와 같이 표현할 수 있습니다.

아래는 예시입니다.

이야기 개요: 민철이는 칼국수를 만들기 위해 면이 필요하다.

아이템 및 위치:
- 냄비는 가스레인지 위에 있다.
- 면은 냉장고 안에 있다.
- 부엌 칼은 테이블 위에 있다.

인물 1:
{ObjectPlacementConstants.NAME}: 민철
{ObjectPlacementConstants.ROLE}: 셰프를 꿈꾸는 사람
{ObjectPlacementConstants.MOTIVE}: 칼국수 한 그릇을 만들고 싶어 한다.

인물 2:
{ObjectPlacementConstants.NAME}: 수민
{ObjectPlacementConstants.ROLE}: 룸메이트
{ObjectPlacementConstants.MOTIVE}: 민철이와 어울리며, 본인도 배가 고프다.

인물 3:
{ObjectPlacementConstants.NAME}: 명진
{ObjectPlacementConstants.ROLE}: 불쑥 찾아온 방문객
{ObjectPlacementConstants.MOTIVE}: 민철이의 친구이며, 배가 고파서 예고 없이 찾아왔다.

출력 예시:
민철이와 수민이는 평온한 저녁을 보내고 있었다. 둘 다 약간 배가 고파지자 민철이가 요즘 연습 중인 칼국수를 먹기로 했다. 모든 사람은 민철이가 "셰프 친구"임을 알고 있었고, 맛있는 음식을 자주 만든다는 것도 알았다. 그래서 예고 없이 방문한 배고픈 친구 명진이가 찾아오게 된 것이었다. 세 사람 모두 냄비가 이미 가스레인지 위에 있다는 것을 알아차렸고, 칼국수를 만들기에 딱 좋다고 생각했다! 부엌 칼이 테이블 위에 있으며, 면은 냉장고 안에 있다는 사실도 알고 있었다.

이제 당신의 차례입니다.

이야기 개요: {story_desc}

아이템 및 위치:
{facts_str}

인물 1:
{ObjectPlacementConstants.NAME}: {people_data[0]['name']}
{ObjectPlacementConstants.ROLE}: {people_data[0]["role"]}
{ObjectPlacementConstants.MOTIVE}: {people_data[0]["motivation"]}

인물 2:
{ObjectPlacementConstants.NAME}: {people_data[1]['name']}
{ObjectPlacementConstants.ROLE}: {people_data[1]["role"]}
{ObjectPlacementConstants.MOTIVE}: {people_data[1]["motivation"]}

인물 3:
{ObjectPlacementConstants.NAME}: {people_data[2]['name']}
{ObjectPlacementConstants.ROLE}: {people_data[2]["role"]}
{ObjectPlacementConstants.MOTIVE}: {people_data[2]["motivation"]}

출력:
""".strip()

                if verbose:
                    print('--- OPENING PROMPT ---')
                    print(opening_prompt)

                opening_output, _ = creator.inference(opening_prompt, model_to_use)

                if verbose:
                    print('=== OPENING OUTPUT ===')
                    print(opening_output)

                chapters.append(opening_output)
                story_so_far += opening_output
            else:
                # For every move, we will generate two "chapters".  A chapter for the actual move happening and a
                # chapter for the observations of everyone else.  We keep these separate to prevent the LLM from saying
                # things like "Claire wasn't able to see X because she was cooking" or things like that.  We would
                # rather the LLM just say "Claire was cooking".

                paragraphs = []

                children = [x for x in n.children if ObjectPlacementConstants.WHEN_MOVING in x.value]
                paras = [x for x in n.children if ObjectPlacementConstants.WHEN_MOVING not in x.value]
                _n = copy.deepcopy(n)
                _n.children = children
                mtree = copy.deepcopy(tree)
                mtree.nodes[0].children = [_n]

                facts = [n.value, *children]  # mtree.get_facts()
                facts_str = "\n".join([f'- {x}' for x in facts])

                try:
                    moving_character = [x for x in people_data if x['name'] == facts[0].split(' ')[0][:-1]][0]
                except Exception as e:
                    print(e)
                    failed_parse = True
                    break

                move_prompt = f"""
우리가 작성한 이야기를 계속 이어서 작성해주세요. 다음에 일어날 이 사건에 대해 짧은 설명을 써 주세요. 오직 이동에 대해서만 작성하고, 다른 추가 정보는 포함하지 마세요. 

절대 "누군가가 뭔가를 보지 못했다"거나, 누군가가 어디에 있는지 추론할 수 있는 능력이 있음을 시사하지 마세요. "몰래" 혹은 "알 수 없게"와 같은 표현도 하지 마세요!
아래는 예시입니다.

한두 문장만 작성하세요. 매우 짧은 설명이어야 합니다.

이야기 개요: 용준이는 자신이 받을 자격이 있다고 생각했던 직업을 상우가 부정행위를 통해 차지했다고 생각하여 화가 났습니다! 그래서 용준이는 상우의 소지품을 버리기 시작했습니다.

인물:
{ObjectPlacementConstants.NAME}: 용준
{ObjectPlacementConstants.ROLE}: 최근 졸업했고 아파트를 공동으로 사용 중인 사람
{ObjectPlacementConstants.MOTIVE}: 부정행위로 룸메이트가 자신의 취업 자리를 빼앗았다고 생각해 매우 화가 나 있다.

이벤트:
- 용준이가 차 열쇠를 쓰레기통으로 옮긴다. 이유: 용준이는 상우에게 화가 나 그의 열쇠를 버리고 싶었다.
- 용준이가 차 열쇠를 옮기다가 쓰레기통에 있는 아이폰을 발견했다.

출력 예시: 분노에 찬 동작으로 열쇠가 깡통 쓰레기통에 부딪히며 소리를 냈다. 곧이어 예상치 못한 *퍽* 소리가 났고... 잠시 호기심이 화를 억누르자, 용준이는 쓰레기통 안에서 아이폰을 발견했다.

또 다른 예시:

이야기 개요: 민수는 방금 새 아파트로 이사했지만, 이전 세입자가 집을 엉망으로 만들어 놓고 떠났다! 집주인은 아무런 조치도 취하지 않아 민수가 직접 청소를 해야 한다.

인물:
{ObjectPlacementConstants.NAME}: 민수
{ObjectPlacementConstants.ROLE}: 새롭지만 지저분한 아파트에 막 이사 온 사람
{ObjectPlacementConstants.MOTIVE}: 이전 세입자가 남긴 어질러진 상태를 혼자 치워야 하며, 관리 사무소는 전혀 도움이 되지 않는다.

이벤트:
- 민수가 국수를 팬트리로 옮긴다. 

이유: 민수는 아파트를 깨끗하게 만들고 싶었고, 국수를 제자리에 두는 것이 마지막 단계였기 때문이다!

출력 예시: 민수는 기쁨이 가득한 표정으로 국수를 팬트리에 넣었다. 끝이 없을 것 같던 쓰레기와 잡동사니 치우기는 드디어 마무리되어, 아파트가 비로소 깔끔해졌다!

당신의 차례입니다. 

이야기 개요: {story_desc}

인물:
{ObjectPlacementConstants.NAME}: {moving_character['name']}
{ObjectPlacementConstants.ROLE}: {moving_character["role"]}
{ObjectPlacementConstants.MOTIVE}: {moving_character["motivation"]}

이벤트:
{facts_str}

출력:
{story_so_far}
""".strip()


                if verbose:
                    print('--- MOVE PROMPT ---')
                    print(move_prompt)

                output, _ = creator.inference(move_prompt, model_to_use)

                if verbose:
                    print('=== MOVE OUTPUT ===')
                    print(output)

                paragraphs.append(output)
                story_so_far += f'\n\n{output}'

                stree = copy.deepcopy(tree)
                stree.nodes = [n]
                stree.nodes[0].children = []

                for p in paras:
                    if len(p.children) > 0:
                        stree.nodes[0].children.extend(p.children)
                    else:
                        stree.nodes[0].children.append(p)

                obs_facts = stree.get_facts()
                obs_facts_str = "\n".join(sorted([f'- {x.value}' for x in obs_facts]))

                obs_prompt_beginning = f"""
지금까지 작성된 이야기를 이어서, 아래에 제시된 관찰 사실들에 대해 작성하세요.
단, 사실들에 대해서만 쓰고 새로운 정보를 추가하지 마세요.
"누군가 보았다", "알아차리지 못했다"라는 표현을 사용하지 말고,
누군가가 무언가를 본 사실이 유일한 관찰 사실일 때만 "누군가가 X를 보았다"라고 쓰세요.

이야기의 분위기를 잡는 데 사용할 수 있는 추가 정보가 있을 수 있지만, 
항상 관찰 사실을 이야기의 주요 가이드로 삼아야 합니다.

아래 이야기의 핵심 물건들은 절대 언급하지 마세요:
{items_str}

{
    "이 단락 이후에 다른 사건이 있을 예정이므로, 사실에만 충실하여 이 단락을 갑작스럽게 마무리하세요. 일반적인 진술은 하지 마세요. 마지막 문장은 제시된 사실에 관한 내용만 담고, 온전한 문장이어야 합니다."
    if loop_idx < len(tree.nodes[0].children) - 1
    else
    "이 이야기는 여기서 끝입니다. 마지막 단락을 마무리하는 문장을 작성하세요."
}

이야기는 현재의 가장 최근 상황에서 벌어지고 있다고 가정하세요.
다시 말해, 아래의 내용:

"{output}"

이 진행되는 동안, 당신이 작성해야 하는 관찰 사실들도 동시에 일어나고 있습니다.

예시를 들어보겠습니다.

이야기 개요: 철수, 유리, 지훈은 하루를 준비 중입니다. 철수는 커리어의 성패가 달린 큰 회의를 앞두고 준비해야 합니다. 
유리는 철수의 회의를 도와주고, 그날 늦게 그 회의 이야기를 듣게 되기를 고대합니다.
지훈은 학교 시험을 준비 중이지만, 아빠인 철수의 중요한 회의에 다소 배려 없는 모습을 보이고 있습니다.

인물 1:
{ObjectPlacementConstants.NAME}: 철수
{ObjectPlacementConstants.ROLE}: 남편
{ObjectPlacementConstants.MOTIVE}: 큰 회의를 앞두고 있고, 그 결과가 자신의 커리어를 좌우할 수 있음.

인물 2:
{ObjectPlacementConstants.NAME}: 유리
{ObjectPlacementConstants.ROLE}: 아내
{ObjectPlacementConstants.MOTIVE}: 언제나 가족을 적극적으로 돕고 싶어 하며, 가족에게 최선을 바라는 인물.

인물 3:
{ObjectPlacementConstants.NAME}: 지훈
{ObjectPlacementConstants.ROLE}: 아들
{ObjectPlacementConstants.MOTIVE}: 곧 학교에서 큰 시험을 앞두고 있어 긴장하고 있으며, 무심코 주변 사람들에게 배려를 덜 하게 됨.

관찰 사실:
- 철수가 아침 식사를 준비하고 있음
- 주방에는 쓰레기통이 없음
- 유리는 밖에서 정원을 물 주며 가꾸고 있음
- 유리는 쓰레기통이 있는 방을 창문으로 볼 수 있음

출력 예시: 철수는 하루를 시작하기 전 배가 고파서 아침을 준비하고 있었다. 그런데 주방에는 쓰레기통이 없었다. 
유리는 언제나 식물을 잘 돌보는 편이라 정원에 물을 주고 있었고, 가까운 창문을 통해 쓰레기통이 있는 방을 볼 수 있었다.

이제 당신의 차례입니다. 

이야기 개요: {story_desc}

인물 1:
{ObjectPlacementConstants.NAME}: {people_data[0]['name']}
{ObjectPlacementConstants.ROLE}: {people_data[0]["role"]}
{ObjectPlacementConstants.MOTIVE}: {people_data[0]["motivation"]}

인물 2:
{ObjectPlacementConstants.NAME}: {people_data[1]['name']}
{ObjectPlacementConstants.ROLE}: {people_data[1]["role"]}
{ObjectPlacementConstants.MOTIVE}: {people_data[1]["motivation"]}

인물 3:
{ObjectPlacementConstants.NAME}: {people_data[2]['name']}
{ObjectPlacementConstants.ROLE}: {people_data[2]["role"]}
{ObjectPlacementConstants.MOTIVE}: {people_data[2]["motivation"]}

관찰 사실:
{obs_facts_str}
                """.strip()

                output_obs_prompt = f'''
출력:
{story_so_far}
                '''

                good_parse = False
                obs_retry = 0
                while obs_retry < 3:
                    obs_retry += 1
                    if verbose:
                        print('--- OBS STORY PROMPT ---')
                        print(f'{obs_prompt_beginning}\n\n{output_obs_prompt}')

                    obs_output, _ = creator.inference(f'{obs_prompt_beginning}\n\n{output_obs_prompt}', model_to_use)

                    if verbose:
                        print('=== OBS STORY OUTPUT ===')
                        print(obs_output)

                    if any([x.lower() in obs_output.lower() for x in items]):
                        obs_prompt_beginning += f"\n\n이전에 생성된 출력 중 하나는 다음과 같았습니다:\n\n{obs_output}\n\n이 출력은 아래 핵심 아이템을 언급했기 때문에 잘못되었습니다:\n{items_str}\n\n다음 번에 작성할 때는 이 핵심 아이템을 언급하지 않도록 유의하세요. 이는 독자에게 혼란을 줄 수 있기 때문입니다."

                        print("FAILED OBS PROMPT")
                    else:
                        good_parse = True
                        break

                if not good_parse:
                    failed_parse = True
                    break

                paragraphs.append(obs_output)
                story_so_far += f' {obs_output}'

                sub_paras = "\n\t".join(paragraphs[1:])
                chapter = f'{paragraphs[0]}\n\t{Conjunctions.BESIDES}, {sub_paras}'
                chapters.append(chapter)
        if failed_parse:
            print("ERROR: FAILED TO PARSE OBS PARAGRAPH.")
            continue

        story = story_so_far

        cost = model_to_use.total_cost + validator_model_early_escape.total_cost + validator_model.total_cost
        total_cost += cost
        print(
            f"EXAMPLE COST: {cost:.2f} | TOTAL COST: {total_cost:.2f}")
        model_to_use.total_cost = 0.0
        validator_model_early_escape.total_cost = 0.0
        validator_model.total_cost = 0.0

        if verbose:
            print("FINISHED")
            print(tree.print_for_gpt())
            print(story)

            print('\n' * 2)

            for idx, q in enumerate(question):
                print(f'{idx + 1}: {q}')
                for l in locations:
                    print(l)


        dataset.append(
            creator.create_dataset_question_object(
                context=story, questions=question, answers=answers,
                intermediate_trees=[[tree]] * len(question), choices=[locations] * len(question),
                intermediate_data=[[{'events': events, 'beliefs': beliefs, 'actual_locs': actual_locs}]] * len(question)
            )
        )

        __idx += 1


    json.dump(dataset, out_file.open('w'), ensure_ascii=False)

    print(f"TOTAL COST: {total_cost} | {total_cost / max_examples} per example.")



if __name__ == "__main__":
    main()