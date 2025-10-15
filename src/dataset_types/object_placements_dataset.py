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



# 이것은 dataset_builder.py에서 원래의 _base_completion_prompt_intro_를 덮어쓰기 위해 사용됩니다.
# 구체적으로, 이는 오브젝트 이동 트리를 생성할 때 사용됩니다.
__object_placements_completion_intro__ = f'''
우리는 사람들이 많은 아이템들을 여러 번 옮기는 {STORY}를 만들고 있습니다. 이 이야기의 목표는 흥미롭고 독창적이면서도, 물건이 어디에 있는지를 명확하게 추적할 수 있도록 하여, 나중에 독자나 언어 모델에게 세계 상태나 누가 무엇을 알고 있는지에 대해 질문할 수 있도록 하는 것입니다.

이 이야기를 만들기 위해, 우리는 이야기 전개와 세계 상태의 변화를 개요화한 트리 구조를 만들었습니다. 당신의 역할은 누군가가 어떤 사건을 보았는지 또는 보지 못했는지를 증명하는 포함 추론 트리를 완성하는 것입니다.

논리 트리는 중간 노드들이 자식 노드들에 의해 포함되는 트리 구조입니다. 이들은 사실들의 집합에 대해 자연어로 된 추론 증명을 만듭니다.

이 트리를 완성하기 위해서는 하나의 함의 추론을 완성해야 합니다. 함의 추론을 완성하는 것은 논리 추론 트리의 한 서브트리를 채우는 것과 유사합니다. 이 단계를 채우기 위해서는 단계의 구조를 따라야 합니다.

"{NodeTypeConstants.FACT_FROM_STORY}"는 우리가 이야기를 쓸 때 명시적으로 서술될 사실들입니다.
"{NodeTypeConstants.COMMONSENSE}"는 대부분의 사람들이 사실로 받아들이는, 명시적으로 말할 필요가 없는 사실들입니다.
"{NodeTypeConstants.COMPLEX_FACT}"는 이야기에서의 더 단순한 사실들에 의해 함의 사실들입니다 (이들은 나중에 재귀 호출을 통해 채워질 것입니다!)

이 단계의 모든 사실은 결합되어 루트 상위 사실을 포함해야 합니다.

현재의 구조 트리와 모순되는 사실은 포함할 수 없습니다.

다른 사람들에 대한 사실은 포함하지 마세요. 무언가가 움직이는 것을 보고 있는지 여부에 대한 그 사람에 대한 사실에만 집중하세요.

'{PromptStep.ENTAILMENT_STEP_TO_COMPLETE_STR}'과 정확히 일치하게 작성하세요. {NodeTypeConstants.FACT_FROM_STORY}와 {NodeTypeConstants.COMMONSENSE}의 개수도 동일하게 작성하고, 순서도 그대로 따라야 합니다.

누군가가 어떤 것을 보았거나 보지 못했다고 명시적으로 말하지 마세요. 당신의 역할은 이를 암시하는 사실들을 제공하는 것입니다. 예를 들어, 지수가 철수가 무언가를 옮기는 것을 보았다고 암시하고 싶다면, "지수는 집안일을 하면서 철수를 보고 있었다" 그리고 "누군가를 보고 있다면, 그 사람이 무엇을 하고 있는지를 보는 것이다"라고 말할 수 있습니다. 누군가가 본 사실을 직접적으로 말하지 않고, 보는 행위의 물리적 조건을 설명하는 방식입니다.

아이템이 움직였다는 사실을 언급하거나 이미 등장한 아이템을 재사용하지 마세요. 예를 들어, "철수가 자신의 아이폰을 보지 못했다"라는 사실이 있고, 당신이 "유림이는 사과가 움직이는 것을 보지 못했다"를 증명하고 있다면, 철수의 아이폰을 다시 사용하지 마세요. 우리 프로그램은 아이템이 어디에 있는지를 엄격하게 제어하고 있으며, 우리가 고려하지 않은 아이템 배치를 도입하는 것을 원하지 않습니다.

각 사실은 추론을 위해 결정적이어야 합니다. 일부러 세부사항을 생략하여 다른 사실들이 이를 보완하도록 하세요. 하나의 사실이 빠지면 결론이 도출되지 않아야 합니다. 같은 사실을 반복해서 사용하지 마세요.

"그"나 "그녀" 같은 대명사 대신 항상 사람의 이름을 사용하세요. 이름을 알고 있다면 반드시 이름을 사용하세요.

한 번에 하나의 추론만 수행하세요. 당신의 추론은 반드시 "{PromptStep.ENTAILMENT_STEP_TO_COMPLETE_STR}" 템플릿과 정확히 일치해야 하며, 우리가 나중에 이를 파싱할 수 있어야 합니다.
'''.strip()



class ObjectPlacementsDataset(DatasetBuilder):
    """
    Wrapper of the datasetbuilder class for specific Object Placements functionality.

    Details specific to Object Placements will be covered here. Details about generic dataset building can be found in
    DatasetBuilder's file.
    """

    def create_sequence_v2(
            self,
            items: List[str],
            locs: List[str],
            people: List[str],
            max_sequence_length: int = 10,
            chance_subject_sees: float = 0.25,
            max_location_use_per_item: int = 2,
            max_movers_per_event: int = 1,
            allowed_movers: List[str] = None,
            initial_starting_positions: List[Tuple[str, str]] = None
    ):
        """
        This code will create a sequence of events where someone moves an item to a new location and other people may
        or may not see that move happen.  We use this to check the validity of a GPT4 sequence.  It's not optimal, but
        it works.


        The complexity of this algo is in making sure someone moves an item only if they've seen the item move to it's
        current location (you can't move something if you don't know where that thing is). As well as keeping track of
        everyones belief states (where people think an item is)

        :param items: List of items people can move
        :param locs: List of locations someone can move items to.
        :param people: List of people who can move stuff
        :param max_sequence_length: Max # of moves
        :param chance_subject_sees: Chance for someone (who is not the mover) to see the item move.
        :param max_location_use_per_item: # of times an item can move.
        :param max_movers_per_event: # of movers per event (usually 1)
        :param allowed_movers: Not used (you can make someone be an observer only)
        :param initial_starting_positions: Where the items are currently.
        """

        def respect_article(item, people):
            if any([item.startswith(name) for name in people]):
                return item
            # return f'the {item}'
            return item

        alive_locs_per_item = {item: deepcopy(locs) for item in items}

        if initial_starting_positions is not None:
            items_to_locations = {
                item: loc for (item, loc) in initial_starting_positions
            }
            for item in items:
                if item in list(items_to_locations.keys()):
                    continue
                else:
                    items_to_locations[item] = random.sample(locs, 1)[0]

            items = list(items_to_locations.keys())

        else:
            items_to_locations = {
                item: loc for item, loc in zip(items, [random.sample(locs, 1)[0] for _ in range(len(items))])
            }

        items_to_people_info = {
            item: {name: {'known_location': True} for name in people} for item in items
        }
        location_histories = {item: {loc: 0 for loc in locs} for item in items}

        # _update = f'{ObjectPlacementConstants.add_josa(item, "objet")} 이동할 때, {ObjectPlacementConstants.add_josa(mover, "name")} {ObjectPlacementConstants.add_josa(i, "object")} {location}에서 보았다.'
        events = [[f'{ObjectPlacementConstants.add_josa(n, "mover")} {ObjectPlacementConstants.add_josa(i, "object")} {l}에서 보았다.' for n in people for (i, l) in items_to_locations.items()]    ]        
        beliefs = [
            {n: {i: l for (i, l) in items_to_locations.items()} for n in people}
        ]
        actual_locs = [deepcopy(items_to_locations)]
        event_structure = [
            {
                'event': '이야기 시작',
                'immutable_sequence': events[0],
                'sequence': []
            }
        ]

        for event_idx in range(max_sequence_length):

            # You can only move items if you know where they are.
            possible_items = {k: [x for x in items if items_to_people_info[x][k]['known_location']] for k in people}

            # A mover can only be a mover if they have something to move.
            possible_movers = [x for x in (people if allowed_movers is None else allowed_movers) if len(possible_items[x]) > 0]

            updates = []

            for mover_idx in range(max_movers_per_event):
                if len(possible_movers) == 0:
                    break
                structure = {
                    'event': '',
                    'immutable_sequence': [],
                    'sequence': []
                }

                mover = random.sample(possible_movers, 1)[0]
                possible_movers.remove(mover)

                item = random.sample(possible_items[mover], 1)[0]

                possible_locations = [x for x in alive_locs_per_item[item] if items_to_locations[item] != x]
                location = random.sample(possible_locations, 1)[0]

                # Mover updates
                _update = ObjectPlacementConstants.move(mover, item, location)
                updates.append(_update)
                structure['event'] = _update

                # The mover now has seen any items at the current spot.
                for i in [x for x in items if items_to_locations[x] == location and not items_to_people_info[x][mover]['known_location']]:
                    items_to_people_info[i][mover]['known_location'] = True
                    _update = f'{ObjectPlacementConstants.add_josa(item, "objet")} 이동할 때, {ObjectPlacementConstants.add_josa(mover, "name")} {ObjectPlacementConstants.add_josa(i, "object")} {location}에서 보았다.'
                    structure['immutable_sequence'].append(_update)
                    updates.append(_update)

                items_to_locations[item] = location
                location_histories[item][location] += 1

                if location_histories[item][location] == max_location_use_per_item:
                    alive_locs_per_item[item].remove(location)

                for subject in [x for x in people if x != mover]:
                    sees_update = random.random() < chance_subject_sees

                    if sees_update:
                        items_to_people_info[item][subject]['known_location'] = True
                        _update = f"{ObjectPlacementConstants.add_josa(subject, 'name')} {ObjectPlacementConstants.add_josa(item, 'object')} {ObjectPlacementConstants.add_josa(location, 'dest')} 이동하는 것을 보았다."
                        updates.append(_update)
                        structure['sequence'].append(_update)
                    else:
                        items_to_people_info[item][subject]['known_location'] = False
                        _update = f"{ObjectPlacementConstants.add_josa(subject, 'name')} {ObjectPlacementConstants.add_josa(item, 'object')} {ObjectPlacementConstants.add_josa(location, 'dest')} 이동하는 것을 보지 않았다."
                        updates.append(_update)
                        structure['sequence'].append(_update)

                event_structure.append(structure)

            if len(updates) == 0:
                continue

            beliefs.append({
                name: {item: items_to_locations[item] if items_to_people_info[item][name]['known_location'] else beliefs[-1][name][item] for item in items} for name in people
            })
            actual_locs.append(deepcopy(items_to_locations))
            events.append(updates)

        return events, beliefs, actual_locs, event_structure

    def generate_end_questions(
            self,
            ending_beliefs: Dict[str, Dict[str, str]],
            people: List[str],
            items: List[str],
            locations: List[str],
            event_structure: List[Dict[str, str]],
    ):
        """
        Generate the questions and the answers to those questions.

        :param ending_beliefs: Where do people think items are.
        :param people: The people who are moving things.
        :param items: The items being moved.
        :param locations: The locations things can move to.
        :param event_structure: The main output of create_sequence_v2
        :return:
        """
        def respect_article(item, people):
            if any([item.startswith(name) for name in people]):
                return item
            # the를 붙여줄 필요가 없음.
            return f'{item}'

        last_moves = []

        for i in items:
            for e in reversed(event_structure):
                if i.lower() in e['event'].lower():
                    for p in people:
                        if p.lower() in e['event'].lower().split(' ')[0]:
                            last_moves.append([p, i])
                            break
                    break

        questions = []
        answers = []
        for x in people:
            for y in items:
                if [x, y] in last_moves:
                    continue
                # q = f'Which location is the most likely place {x} would look to find {respect_article(y.lower(), people)} given the story?'
                xs = x.strip()
                ys = y.strip()
                q = f'주어진 이야기를 고려할 때, {ObjectPlacementConstants.add_josa(xs, "name")} {ObjectPlacementConstants.add_josa(ys, "object")} 찾기 위해 어디를 가장 먼저 확인하겠는가?'
                questions.append(q)
                answers.append(locations.index(ending_beliefs[x][y]))
        return questions, answers

    def create_event_trees(
            self,
            model: OpenAIModel,
            event_structure: List[Dict[str, Union[str, List[str]]]],
            items: List[str],
            locations: List[str],
            completion_description: str,
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

        :param model: See datasetbuilder
        :param event_structure: Main output of create_sequence_v2
        :param items: items that can move
        :param locations: Locations items can move to
        :param completion_description: Description we use that is super close to the entailment step (we found this
            helpful for making the trees include more story specific details rather than bland move details.)
        :param description: General description of the story.
        :param example_completion_trees: See datasetbuilder
        :param example_completion_nodes: See datasetbuilder
        :param example_completion_descriptions:  See datasetbuilder
        :param depth: See LogicTree
        :param bf_factor: See LogicTree
        :param chance_to_prune_all: See LogicTree
        :param chance_to_prune: See LogicTree
        :param max_retries_on_error: See datasetbuilder
        :param progress_bar: See datasetbuilder
        :param test_complete_structure_prompt: See datasetbuilder
        :param retry_model: See datasetbuilder
        :param use_complex_facts: See datasetbuilder
        :param use_validators: See datasetbuilder
        """

        nodes = []
        for e in event_structure:

            # usually the beginning of the story (where everything initially is, is immutable and we don't want to add deductions to it)
            children = [
                LogicNode(x, prunable=False, can_be_leaf=True, frozen=True) for x in e['immutable_sequence']
            ]
            # For each move, however, we want to add deductions to why or why not people saw the move.
            children = [*children, *[LogicNode(x) for x in e['sequence']]]
            nodes.append(LogicNode(e['event'], children, frozen=True, prunable=False))

        validators = [StructureValidator(),]
        if use_validators:
            validators.append(ForbiddenTextValidator(
                forbidden_words=[
                    *items, *locations,
                ],
                reason_why="우리는 독자가 물건이 어디에 있는지, 그리고 다른 사람이 물건이 움직이는 것을 보았는지를 추적해야 하는 이야기를 만들고 있습니다. 그렇기 때문에, 어떤 사람의 관찰에 대한 사실을 만들 때, 다른 사람들이 무엇을 하고 있는지, 다른 물건이 어디에 있는지, 또는 다른 장소에 대한 추론은 하지 않으려고 합니다. 이러한 모든 요소들은 명확한 정답이 존재하는 설정을 만들기 위해 명시적으로 제어되고 있습니다."
            ))

        tree = self.complete_structure(
            self.build_structure(
                depth=depth,
                bf_factor=bf_factor,
                chance_to_prune_all=chance_to_prune_all,
                chance_to_prune=chance_to_prune,
                root_nodes=[LogicNode(description, nodes)]
            ),
            model,
            description=completion_description,
            completion_prompt_fn=self.create_completion_prompt(example_completion_trees, example_completion_nodes,
                                                               example_completion_descriptions,
                                                               intro=__object_placements_completion_intro__, because_clause_after=0,
                                                               because_clause='왜냐하면, 이 야야기에 따르면,',
                                                               use_complex_facts=use_complex_facts),
            max_retries_on_error=max_retries_on_error,
            inplace=True,
            progress_bar=progress_bar,
            retry_model=retry_model,
            test_prompt=test_complete_structure_prompt,
            validators=validators,
            use_iterative_complete_v2=use_validators
        )

        return tree
