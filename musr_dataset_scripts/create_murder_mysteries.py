"""
RUN THIS FILE TO CREATE AWESOME MURDER MYSTERIES USING AN LLM :)

Go to the main() function for arguments/control over the dataset creation.

NOTE: Expects your openai api key to be in the environment.  "OPENAI_API_KEY=api_key python script.py" (if you are using
openai LLMs)

NOTE: By default, datasets go into "{ROOT_FOLDER}/datasets/{dataset_name}.json"
"""

import copy
import json
import sys
from pathlib import Path
import random

from src import cache, set_seed

from src import cache
from src.model import OpenAIModel
from src.logic_tree.tree import LogicTree, LogicNode, LogicNodeFactType
from src.madlib.madlib import Madlib
from src.utils.paths import OUTPUT_FOLDER, ROOT_FOLDER
from src.utils.constants import MysteryConstants, PromptStep

from src.dataset_types.murder_mystery_dataset import MurderMysteryDataset
from argparse import ArgumentParser

"""ICL EXAMPLES for creating deductions. See datasetbuilder for more info."""

minsu = "민수"
jimin = "지민"
minchul = "민철"

example1_description = f"""
{MysteryConstants.VICTIM}: 유리
{MysteryConstants.PLACE}: 집
{MysteryConstants.TOOL}: 총
{MysteryConstants.SUSPECT}: 민수
{MysteryConstants.ROLE}: 형제
{MysteryConstants.MOTIVE}: 금전적 이익
"""

example1_tree = LogicTree(
    nodes=[
        LogicNode(MysteryConstants.is_murderer(minsu), [
            LogicNode(MysteryConstants.has_means(minsu), [
                LogicNode("민수는 총 쏘는 연습을 했다."),
                LogicNode("민수는 총을 소유하고 있다."),
                LogicNode(
                    f"총을 소유하고 있으며 사용법을 연습했다면, 살인을 저지를 능력이 있다.",
                    fact_type=LogicNodeFactType.COMMONSENSE
                )
            ]),
            LogicNode(MysteryConstants.has_motive(minsu), [
                LogicNode("민수는 극도로 돈에 절박했다."),
                LogicNode("민수는 유리의 돈에 극도로 절박했다."),
                LogicNode(
                    "누군가가 극도로 절박할 때, 목표를 이루기 위해 극단적인 방법을 사용할 수 있으며, 여기에는 살인도 포함될 수 있다.",
                    fact_type=LogicNodeFactType.COMMONSENSE
                )
            ]),
            LogicNode(MysteryConstants.has_opportunity(minsu))
        ])
    ],
    prune=False, populate=False
)

example1_node_completion = LogicNode(MysteryConstants.has_opportunity(minsu), [
    LogicNode("민수는 유리의 집에 출입할 수 있다."),
    LogicNode(
        f"누군가의 집에 출입할 수 있다는 것은 그들을 살해할 {MysteryConstants.OPPORTUNITY}를 제공한다.",
        fact_type=LogicNodeFactType.COMMONSENSE
    )
])

example2_description = f"""
{MysteryConstants.VICTIM}: 철수
{MysteryConstants.PLACE}: 경마장
{MysteryConstants.TOOL}: 삽
{MysteryConstants.SUSPECT}: 지민
{MysteryConstants.ROLE}: 러닝 파트너
{MysteryConstants.MOTIVE}: 다른 사람에게 해가 가는 것을 막기 위해
"""

example2_tree = LogicTree(
    nodes=[
        LogicNode(MysteryConstants.is_murderer(jimin), [
            LogicNode(MysteryConstants.has_means(jimin), [
                LogicNode("지민이는 농부다"),
                LogicNode(
                    "농부들은 일반적으로 삽과 같은 원예 도구를 사용한다.",
                    fact_type=LogicNodeFactType.COMMONSENSE
                )
            ]),
            LogicNode(MysteryConstants.has_motive(jimin)),
            LogicNode(MysteryConstants.has_opportunity(jimin))
        ])
    ],
    prune=False, populate=False
)

example2_node_completion = LogicNode(MysteryConstants.has_motive(jimin), [
    LogicNode("지민이는 민수를 깊이 사랑한다."),
    LogicNode("철수는 민수를 위협했다."),
    LogicNode(
        "깊고 강렬한 사랑은 사랑하는 사람이 위협받을 때 극단적인 행동, 심지어 살인까지 저지르게 할 수 있다.",
        fact_type=LogicNodeFactType.COMMONSENSE
    )
])

example3_description = f"""
{MysteryConstants.VICTIM}: 준호
{MysteryConstants.PLACE}: 공원 벤치
{MysteryConstants.TOOL}: 헤로인 과다복용
{MysteryConstants.SUSPECT}: 민철
{MysteryConstants.ROLE}: 마약 사용자
{MysteryConstants.MOTIVE}: 공개적인 굴욕
"""

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
        "헤로인에 접근할 수 있고, 치명적인 양을 알고 있다면, 피해자에게 의도적으로 치명적인 용량을 투여하여 살인을 저지를 수 있다.",
        fact_type=LogicNodeFactType.COMMONSENSE
    )
])

example_trees = [example1_tree, example2_tree, example3_tree]
example_node_completions = [example1_node_completion, example2_node_completion, example3_node_completion]
example_descriptions = [example1_description, example2_description, example3_description]


def main():
    parser = ArgumentParser()
    parser.add_argument('--cache', action='store_true', help='Enable caching')
    parser.add_argument("--model", type=str, default='gpt-4o', help="Model to use", choices=["gpt-4o-mini", "o1", "o3-mini", "gpt-4o", "gpt-4"])
    parser.add_argument("--seed", type=int, default=500, help="Random seed")
    parser.add_argument("--num_samples", type=int, default=1, help="Maximum number of examples to generate")
    parser.add_argument("--validation_tries", type=int, default=3, help="Number of validation tries for each chapter generation")
    args = parser.parse_args()

    one_million= 1000000
    gpt35 = OpenAIModel(engine='gpt-3.5-turbo', api_endpoint='chat', api_max_attempts=30, temperature=1.0, max_tokens=1500, num_samples=1, prompt_cost=0.0015/1000, completion_cost=0.002/1000)
    gpt16k35 = OpenAIModel(engine='gpt-3.5-turbo-16k', api_endpoint='chat', api_max_attempts=30, temperature=1.0, max_tokens=2400, num_samples=1, prompt_cost=0.003/1000, completion_cost=0.004/1000)
    gpt4 = OpenAIModel(engine='gpt-4', api_max_attempts=30, api_endpoint='chat', temperature=1.0, top_p=1.0, max_tokens=2400, num_samples=1, prompt_cost=0.03/1000, completion_cost=0.06/1000)
    gpt4omini = OpenAIModel(engine='gpt-4o-mini', api_max_attempts=30, api_endpoint='chat', temperature=1.0, top_p=1.0, max_tokens=2400, num_samples=1, prompt_cost=0.15/one_million, completion_cost=0.6/one_million)
    gpt4o = OpenAIModel(engine='gpt-4o', api_max_attempts=30, api_endpoint='chat', temperature=1.0, top_p=1.0, max_tokens=2400, num_samples=1, prompt_cost=2.5/one_million, completion_cost=10/one_million)
    o1 = OpenAIModel(engine='o1', api_max_attempts=30, api_endpoint='chat', temperature=1.0, top_p=1.0, max_tokens=2400, num_samples=1, prompt_cost=15/one_million, completion_cost=60/one_million)
    o3_mini = OpenAIModel(engine='o3-mini', api_max_attempts=30, api_endpoint='chat', temperature=1.0, top_p=1.0, max_tokens=2400, num_samples=1, prompt_cost=1.1/one_million, completion_cost=4.4/one_million)

    set_seed(args.seed)
    
    # PARAMS (if not with a comment, look at the Murder Mystery dataset class for more info.)

    
    tree_depth = 4

    max_number_of_suspects = 2
    max_structure_completion_retries = 3
    max_num_suspicious_facts = 1

    use_validators = True

    max_examples = args.num_samples
    out_file = OUTPUT_FOLDER / f'custom_murder_mysteries_{args.model}' / f'{args.seed}.json'
    if out_file:
        out_file.parent.mkdir(exist_ok=True, parents=True)
   
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

    validation_tries = args.validation_tries

    
    creator = MurderMysteryDataset()

    dataset = []

    total_cost = 0

    madlib = Madlib(
        {
            "male_names": ROOT_FOLDER / 'domain_seed/male_names.json',
            "female_names": ROOT_FOLDER / 'domain_seed/female_names.json',
            "male_relationships": ROOT_FOLDER / 'domain_seed/male_relationships.json',
            "female_relationships": ROOT_FOLDER / 'domain_seed/female_relationships.json',
            "motives": ROOT_FOLDER / 'domain_seed/strong_motives.json',
            "murder_weapons": ROOT_FOLDER / 'domain_seed/murder_weapons.json',
            "relationships": ROOT_FOLDER / 'domain_seed/relationships.json',
            "crime_scenes": ROOT_FOLDER / 'domain_seed/crime_scenes.json',
            'red_herrings': ROOT_FOLDER / 'domain_seed/suspicious_facts.json',
        }
    )

    previously_sampled_items = []

    # Resume from a previous run. (should be a Path object)
    resume_file = None
    if resume_file and resume_file.exists():
        data = json.load(resume_file.open('r'))
        previously_sampled_items.extend([[x["victim"], x["crime_scene"], x["murder_weapon"]] for y in data for x in [y['questions'][0]['intermediate_data'][0]['victim_info']]])
        dataset = data

    # CREATION LOGIC
    accepted = 0
    attempts = 0
    while accepted < max_examples:
        attempts += 1
        print(f"시도: {attempts}, 완료 예제수: {accepted}/{max_examples}")

        # Setup Scenario (MadLib)
        constant_sampled_items = [['male_names', 'female_names'], 'crime_scenes', 'murder_weapons']
        constant_sampled_names = ['victim', 'crime_scene', 'murder_weapon']
        variable_sampled_items = [['male_names,male_relationships', 'female_names,female_relationships'], 'motives', 'crime_scenes']
        variable_sampled_names = ['suspect', 'role', 'motive', 'alibi']

        description_string = f"{MysteryConstants.VICTIM}: {{victim}}\n{MysteryConstants.PLACE}: {{crime_scene}}\n{MysteryConstants.TOOL}: {{murder_weapon}}"
        variable_string = f"{MysteryConstants.SUSPECT}: {{suspect}}\n{MysteryConstants.ROLE}: {{role}}\n{MysteryConstants.MOTIVE}: {{motive}}"

        victim_string, victim_dict, sampled = creator.sample_madlib(madlib, constant_sampled_items, previously_sampled=previously_sampled_items, description_string_format=description_string, sampled_item_names=constant_sampled_names)
        victim_dict = victim_dict[0]
        previously_sampled_items = sampled

        suspect_strings, suspect_dicts, _ = creator.sample_madlib(madlib, variable_sampled_items, previously_sampled=[[None,None,None,victim_dict['crime_scene']]], n_samples=max_number_of_suspects, description_string_format=variable_string, sampled_item_names=variable_sampled_names)

        _, suspicious_fact_dicts, _ = creator.sample_madlib(madlib, ['red_herrings'], n_samples=max_num_suspicious_facts * len(suspect_dicts), description_string_format='{red_herrings}')
        random.shuffle(suspicious_fact_dicts)
        for s in suspect_dicts:
            s['red_herrings'] = []
            for n in range(max_num_suspicious_facts):
                s['red_herrings'].append(suspicious_fact_dicts.pop()['red_herrings'])

        scenario = f'{victim_string[0]}\n'
        d = f'{scenario}'.strip()
        for idx, s in enumerate(suspect_strings):
            suspect_dicts[idx]['description'] = f"{scenario}{s}".strip()
            d += f"\n\n{s}\n{MysteryConstants.RED_HERRINGS}: {suspect_dicts[idx]['red_herrings'][0]}"

        # Victim dict should have the victim name, crime scene, and murder weapon (things specific to the victim)
        # Suspect dicts should have the the name of the victim, their role in the story, their suspicious fact and motive.

        suspect_trees = creator.create_suspect_trees(
            model_to_use,
            victim_dict,
            suspect_dicts,
            example_trees,
            example_node_completions,
            example_descriptions,
            depth=tree_depth,
            bf_factor={2: 1.0},
            chance_to_prune=0.0,
            chance_to_prune_all=0.0,
            max_num_of_suspicious_facts=max_num_suspicious_facts,
            max_retries_on_error=max_structure_completion_retries,
            retry_model=model_to_use,
            progress_bar=True,
            use_validators=use_validators,
            # model_validator_model=gpt4,
            # model_validator_early_escape_model=gpt16k35,
            model_validator_model=validator_model,
            model_validator_early_escape_model=validator_model_early_escape,
            test_completion_prompt=False
        )
        if not suspect_trees:
            print("용의자 트리 구축 실패. 처음부터 다시 시작.")
            continue

        suspect_trees = creator.create_chapter_trees(suspect_trees, max_num_of_suspicious_facts=max_num_suspicious_facts)

        suspect_trees = creator.create_chapter(model_to_use, suspect_trees, validate_model=model_to_use, validation_tries=validation_tries)

        # Because we only created chapters of the murder, we need an introduction to it.  Here we create a prompt to do that.
        sus_strings = ", ".join([x['suspect_info']['suspect'] for x in suspect_trees])
        intro_prompt = f"살인 미스터리에 대한 도입부를 작성하세요. 도입부는 1~2문장으로만 구성되어야 합니다. 도입부에는 피해자 이름, 피해자가 살해된 장소, 살해에 사용된 도구, 두 명의 용의자 이름을 모두 언급하세요. 도입부 외에는 아무것도 작성하지 마세요. \n\n{PromptStep.SCENARIO}\n{victim_dict['victim']}(이)가 {victim_dict['crime_scene']}에서 {victim_dict['murder_weapon']}에 의해 살해당했습니다. 김철수 형사가 사건을 조사하며 용의자들을 심문하고 있습니다. 용의자는 {sus_strings}입니다.\n\n출력:\n"
        intro, _ = creator.inference(intro_prompt, model_to_use)

        # Iterate through the suspects (the curr suspect is the murderer)
        for sidx in range(len(suspect_trees)):
            murderer_idx = sidx

            _suspect_trees = copy.deepcopy(suspect_trees)

            for sidx, s in enumerate(_suspect_trees):
                _suspect_trees[sidx]['used_chapter'] = _suspect_trees[sidx][
                    'innocent_chapter' if sidx != murderer_idx else 'murderer_chapter']
                _suspect_trees[sidx]['used_tree'] = _suspect_trees[sidx][
                    'innocent_tree' if sidx != murderer_idx else 'murderer_tree']
                _suspect_trees[sidx]['is_murderer'] = sidx == murderer_idx

            chapters = [(x['suspect_info']['suspect'], x['used_chapter'].strip()) for x in _suspect_trees]
            random.shuffle(chapters)

            story = f"{intro}\n\n" + "\n\n".join([x[1] for x in chapters])
            choices = [x['suspect_info']["suspect"] for x in _suspect_trees]

            call_cost = model_to_use.total_cost + validator_model.total_cost + validator_model_early_escape.total_cost
            total_cost += call_cost
            print(f'EXAMPLE COST: {call_cost:.2f} | TOTAL COST SO FAR: {total_cost:.2f}')
            model_to_use.total_cost = 0.0
            validator_model.total_cost = 0.0
            validator_model_early_escape.total_cost = 0.0

            safe_suspects_dict = [{k: v.to_json() if isinstance(v, LogicTree) else v for k, v in x.items()} for x in _suspect_trees]
            dataset.append(
                creator.create_dataset_question_object(
                    context=story,
                    questions=['다음 중 살인자일 가능성이 가장 높은 사람은 누구인가?'],
                    answers=[murderer_idx],
                    choices=[choices],
                    intermediate_trees=[[x['used_tree'] for x in _suspect_trees]],
                    intermediate_data=[[{'suspect_info': safe_suspects_dict, 'victim_info': victim_dict, 'story_hash_id': hash(intro)}]]
                )
            )

        if out_file:
            json.dump(dataset, out_file.open('w'), ensure_ascii=False)
        accepted += 1

    if out_file:
        json.dump(dataset, out_file.open('w'), ensure_ascii=False)

    print(f"TOTAL COST: {total_cost} | {total_cost / max_examples} per example.")


if __name__ == "__main__":
    main()