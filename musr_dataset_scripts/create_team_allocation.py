"""
이 파일을 실행하면 LLM을 사용해 멋진 이야기를 생성할 수 있습니다 :)

데이터셋 생성에 대한 파라미터와 설정을 변경하려면 main() 함수를 참고하세요.

주의: OpenAI LLM을 사용하려면, 환경 변수에 OpenAI API 키가 설정되어 있어야 합니다.
예시: "OPENAI_API_KEY=api_key python script.py"

주의: 기본적으로 생성되는 데이터셋은 "{ROOT_FOLDER}/datasets/{dataset_name}.json" 경로에 저장됩니다.
"""

import pprint

from jsonlines import jsonlines
import json
import sys
from pathlib import Path
import random
from functools import partial



from src import cache,set_seed
from src.model import OpenAIModel
from src.logic_tree.tree import LogicTree, LogicNode, LogicNodeFactType
from src.madlib.madlib import Madlib
from src.utils.paths import OUTPUT_FOLDER , ROOT_FOLDER
from src.utils.constants import MysteryConstants, TeamAllocationConstants
from src.dataset_types.team_allocation import TeamAllocationDataset

import random
from itertools import combinations
from argparse import ArgumentParser


cLogicNode = partial(LogicNode, fact_type=LogicNodeFactType.COMMONSENSE)

"""추론을 만들기 위한 ICL 예시입니다. 자세한 내용은 datasetbuilder를 참조하세요."""

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
            LogicNode(MysteryConstants.has_opportunity(minchul)),
            LogicNode(MysteryConstants.has_opportunity(minchul)),
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


example_descriptions = [example1_description, example2_description, example3_description]
example_trees = [example1_tree, example2_tree, example3_tree]
example_node_completions = [example1_node_completion_tree.children[0].children[0], example3_node_completion, example1_node_completion_tree.children[0]]


def main():
    parser = ArgumentParser()
    parser.add_argument('--cache', action='store_true', help='Enable caching')
    parser.add_argument("--model", type=str, default='gpt-4o-mini', help="Model to use", choices=["gpt-4o-mini", "o1", "o3-mini", "gpt-4o", "gpt-4"])
    parser.add_argument("--seed", type=int, default=500, help="Random seed")
    parser.add_argument("--num_samples", type=int, default=1, help="Maximum number of examples to generate")
    args = parser.parse_args()

    creator = TeamAllocationDataset()

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
    validator_model_early_escape = gpt4omini

    max_examples = args.num_samples
    out_file = OUTPUT_FOLDER / f'custom_team_allocation_{args.model}' / f'{args.seed}.json'

    tree_depth = 2

    structure_retries = 1

    verbose = False

    dataset = []
    previous_samples = [['']]

    # 스토리를 만들기 위한 설명/시나리오를 샘플링
    
    madlib_path = OUTPUT_FOLDER / 'domain_seed' / f'ta_scenario_descriptions.json'
    madlib = Madlib({"scenario_descriptions": Path(madlib_path)})

    total_cost = 0.0

    idx = 0
    max_idx = int(max_examples * 2)
    real_idx = 0
    while idx < max_examples and real_idx < max_idx:
        real_idx += 1
        print(f"예시 {idx + 1} 생성 중...")

        descriptions, _, previous_samples = creator.sample_madlib(madlib, ['scenario_descriptions'],
                                                                  '{scenario_descriptions}',
                                                                  previously_sampled=previous_samples
                                                                  )
        description = descriptions[0]
        
        # 시나리오 설명을 바탕으로 세 명의 사람, 두 개의 작업, 두 개의 기술명을 생성
        prompt = f'''
주어진 시나리오 설명을 바탕으로 세 명의 사람, 두 개의 역할, 두 개의 기술을 만들어주세요.
각 기술은 서로 독립적이어야 하고, 각각의 업무에 대응되어야 합니다.

규칙:
 1) 이름에 특정 직함이나 기술, 업무 능력을 암시하지 마세요. 예: "김 박사"처럼 쓰지 마세요.

예시:
설명: 많은 손님들이 커피바에 몰려옵니다. 당신은 몇 명을 계산대로, 몇 명을 바리스타로 배치해야 합니다.

출력:

사람들: 준호; 민철; 정윤
역할: 바리스타; 계산원
능력: 커피를 만들 수 있음; 고객들을 응대할 수 있음

당신의 차례!

설명: {description}

출력:
        '''.strip()
        output, _ = creator.inference(prompt, model_to_use)

        cost = model_to_use.total_cost + validator_model.total_cost + validator_model_early_escape.total_cost
        total_cost += cost
        model_to_use.total_cost = 0.0
        validator_model.total_cost = 0.0
        validator_model_early_escape.total_cost = 0.0

        people = []
        skills = []
        tasks = []

        # 출력 문자열에서 사람, 업무, 기술 리스트를 파싱
        try:
            for line in output.split('\n'):
                if line.startswith('사람들:'):
                    people.extend([x.strip() for x in line.replace('사람들:', '').strip().split(';') if x != ''])
                elif line.startswith('역할:'):
                    tasks.extend([x.strip() for x in line.replace('역할:', '').strip().split(';') if x != ''])
                elif line.startswith('능력:'):
                    skills.extend([x.strip() for x in line.replace('능력:', '').strip().split(';') if x != ''])

                    

        except Exception as e:
            print("에러")
            print(e)
            continue

        else: 
            for line in output.split('\n'):
                if not line.startswith(("사람들:", "역할:", "능력:")):
                    continue 
        

        # 각 사람에 대한 기술/업무 관계를 임의로 생성하고 사실 집합(F)을 만든다.
        people_levels, best_pair, all_pairs = creator.build_assignment(people)
        print(f"사람들: {people}, 업무: {tasks}, 기술: {skills}"
              f", 사람 레벨: {people_levels}, 최적 조합: {best_pair}, 모든 조합: {all_pairs}")
        
        
        facts = creator.create_facts(people_levels, people, tasks)

        random.shuffle(facts)
        random.shuffle(all_pairs)

        # 트리를 생성 (ICL 예시에 기반하여 트리 구조를 확장)
        tree = creator.create_fact_trees(
            model_to_use, facts, tasks, description, example_trees, example_node_completions,
            example_descriptions, depth=tree_depth, bf_factor={2:1.0}, chance_to_prune=0.0, chance_to_prune_all=0.0,
            max_retries_on_error=structure_retries, progress_bar=True, test_complete_structure_prompt=False,
            retry_model=model_to_use
        )

        def fact_str(t):
            facts = list(sorted(list(set([x.value for x in t.get_facts()]))))
            facts_str = "\n".join([f'- {x}' for x in facts])
            return facts_str

        facts = fact_str(tree)

        # 스토리 생성. 사실들을 스토리에 모두 포함해야 함.
        prompt = f'''
당신은 아래 시나리오 설명과 사실들의 목록을 보고 짧은 이야기를 작성할 것입니다.
이야기 속에서 모든 사실을 포함해야 합니다.

당신은 매니저나 리더의 역할을 맡고 있으며, 사람들을 업무와 기술에 배정해야 합니다.

하지만 어떤 배정이 '가장 좋은' 배정인지 답을 제시하지 않고, 독자가 그 답을 고민하도록 남겨두세요.
즉, 누구에게 어떤 업무를 맡길지 결정하지 말고, 이야기 형식으로 서술만 해주세요.

이야기는 흥미롭고, 일관성 있게 작성해 주세요.

도입부를 먼저 쓰고, 세 명의 사람을 소개하세요:
- {people[0]}
- {people[1]}
- {people[2]}

그리고 매니저가 이들을 배치해야 하는 두 개의 업무도 초반에 언급하세요:
- {tasks[0]}
- {tasks[1]}

설명: {description}
아래 사실들을 반드시 이야기 속에 포함하세요:
{facts}

출력:
        '''.strip()

        if verbose:
            print("--- MAIN STORY PROMPT ---")
            print(prompt)
        output, _ = creator.inference(prompt, model_to_use)
        if verbose:
            print("=== MAIN STORY OUT ===")
            print(output)

        paragraphs = output.split('\n\n')

        # 이야기가 제대로 된 도입부를 갖추도록 도입부를 다시 다듬는 프롬프트
        fix_p0_prompt = f'''
내가 작성한 이야기는 다음과 같습니다:

{output}

---

위 이야기의 도입부를, 사람들이나 업무를 배정하지 않고(이것은 독자가 고민해야 할 내용), 
세 명의 사람(이름만 간단히 언급)과 두 개의 업무 이름을 자연스럽게 소개하도록 다시 작성해주세요.

아래는 현재 도입부입니다:
{paragraphs[0]}

언급해야 할 세 인물입니다:
- {people[0]}
- {people[1]}
- {people[2]}

언급해야 할 두 개의 업무 이름입니다:
- {tasks[0]}
- {tasks[1]}

도입부는 짧게, 원본 도입부보다 길지 않게 작성해주세요.
        '''.strip()

        if verbose:
            print('--- FIX P0 PROMPT ---')
            print(fix_p0_prompt)

        paragraphs[0], _ = creator.inference(fix_p0_prompt, model_to_use, temperature=0.2)

        if verbose:
            print("=== FIXED P0 ===")
            print(paragraphs[0])

        output = "\n\n".join(paragraphs)

        question = "이 이야기를 바탕으로, 두 가지 업무가 모두 효율적으로 수행되도록 사람을 어떻게 배정하시겠습니까?"

        choices = [
            f"{tasks[0]}: {all_pairs[0][0][0]} - {tasks[1]}: {all_pairs[0][1][0]}, {all_pairs[0][1][1]}",
            f"{tasks[0]}: {all_pairs[1][0][0]} - {tasks[1]}: {all_pairs[1][1][0]}, {all_pairs[1][1][1]}",
            f"{tasks[0]}: {all_pairs[2][0][0]} - {tasks[1]}: {all_pairs[2][1][0]}, {all_pairs[2][1][1]}",
        ]

        if verbose:
            print(f"예시 {idx + 1} 최종 출력:")
            print(output)
            print('\n\n')
            print(question)
            print('\n\n')
            print("\n".join([f'{cidx+1} - {x}' for cidx, x in enumerate(choices)]))

        gold_idx = all_pairs.index(best_pair)

        idx += 1
        dataset.append(
            creator.create_dataset_question_object(
                output,
                questions=[question],
                answers=[gold_idx],
                choices=[choices],
                intermediate_trees=[[tree]],
                intermediate_data=[[{'tasks': tasks, 'matrix': people_levels, 'best_pair': best_pair}]]
            )
        )

        cost = model_to_use.total_cost + validator_model.total_cost + validator_model_early_escape.total_cost
        total_cost += cost
        gpt4.total_cost = 0.0
        gpt35.total_cost = 0.0
        gpt16k35.total_cost = 0.0

        print(f"예시 하나 당 비용: {cost} | 지금까지의 총 비용 {total_cost}")

    out_file.parent.mkdir(exist_ok=True, parents=True)
    json.dump(dataset, out_file.open('w'), ensure_ascii=False)


if __name__ == "__main__":
    main()