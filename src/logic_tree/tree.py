"""
경고 (혹은 다소 강한 주의사항).

이전 실험을 위해 많은 기능이 이곳에 구현되었습니다. 대부분은 현재 사용되지 않습니다. 
그러나 현재 데이터셋과의 하위 호환성을 유지하기 위해 남겨두었으며, 그냥 둬도 문제될 것이 없다고 판단했습니다.

또한 참고할 사항:

이 파일은 리포지토리 내 어떤 것도 의존하지 않도록 설계되었습니다. 
이 파일을 복사하여 자신의 프로젝트에서 데이터셋의 논리 트리를 파싱/시각화/편집하거나 
자신만의 논리 트리를 만들 수 있습니다.

마지막 참고 사항:

파일의 __main__ 부분에서 LogicNode와 LogicTree를 생성하는 예제를 확인하세요.
"""

import random
from typing import List, Any, Dict
from enum import Enum
import numpy as np
from copy import deepcopy

from src.utils.constants import NodeTypeConstants


class LogicNodeOperatorType:
    """추론이 노드를 어떻게 결합해야 하는가 (choose는 populate가 호출될 때 무작위로 샘플링됨)"""
    AND = '그리고'
    OR = '또는'
    CHOOSE = '선택'


class LogicNodeFactType:
    """노드가 명시적인 것인가 (스토리에서 직접 언급됨) 아니면 상식적인 지식인가 (암시됨)"""
    EXPLICIT = '사실'
    COMMONSENSE = '상식'


class LogicNodeConstraints:
    """ 
    유용한 제약 조건 예시: 
    children = ['X가 살인자이다', 'Y가 살인자이다', 'Z가 살인자이다']
    하지만 이 구조는 더 이상 사용되지 않음.
    """
    ONLY_ONE_CAN_BE_TRUE = '오직 하나의 자식만 참일 수 있음'


class LogicNodeDeductionType:
    """여기에 어떤 유형의 추론을 사용할 것인가 (현재 사용되지 않음)"""
    SYLLOGISM = '삼단 논법'
    TEMPORAL = '시간적'
    SPATIAL = '공간적'
    CHOOSE = '선택'


class LogicNode:
    """
    LogicNode는 트리의 기본 단위입니다. 
    추론 노드이거나, 리프 사실 노드일 수 있습니다. 
    리프 사실은 (명시적인 경우) 스토리 생성에서 사용됩니다.
    """
    value: str
    children: List['LogicNode']
    fact_type: str
    operator: str
    constraints: List[str]
    deduction_type: str
    prunable: bool
    can_be_leaf: bool

    def __init__(
            self,
            value: str = '',
            children: List['LogicNode'] = None,
            operator: str = LogicNodeOperatorType.OR,
            fact_type: str = LogicNodeFactType.EXPLICIT,
            constraints: List[str] = (),
            deduction_type: str = None,
            prunable: bool = True,
            can_be_leaf: bool = False,
            frozen: bool = False,
    ):
        """
        :param value: 이 특정 노드의 내용 (또한 자식들의 추론 결과).
        :param children: 이 노드의 자식 노드들.
        :param operator: 자식 노드들이 "AND" 또는 "OR"로 결합되어 이 노드의 추론 결과를 형성할지 여부.
        :param fact_type: 명시적 사실인지 상식인지.
        :param constraints: 더 이상 사용되지 않음 (LogicNodeConstraints 참조).
        :param deduction_type: 더 이상 사용되지 않음 (LogicNodeDeductionType 참조).
        :param prunable: 이 노드를 트리에서 제거할 수 있는지 여부 (현재 데이터셋에서는 가지치기를 하지 않음).
        :param can_be_leaf: 이 노드가 리프 노드가 될 수 있는지 여부 (일반적으로 수동으로 삽입하는 노드는 false).
        :param frozen: populate 함수에서 자식 노드를 추가하거나 제거해야 하는지 여부.
                       (frozen이 true이면 자식이 추가/삭제되지 않지만, 자식들의 자식은 변할 수 있음).
        """
        self.value = value
        if children is None:
            children = []
        self.children = children
        self.operator = operator
        self.fact_type = fact_type
        self.constraints = constraints
        self.deduction_type = deduction_type
        self.prunable = prunable
        self.can_be_leaf = can_be_leaf
        self.frozen = frozen
        self.parent = None

    @property
    def children(self):
        return self._children

    @children.setter
    def children(self, children: List['LogicNode']):
        self._children = children
        for c in self.children:
            c.parent = self

    def __str__(self):
        line = []
        cnsts = ', '.join([str(x.value) for x in self.constraints])

        if self.value and self.value != '':
            line.append(self.value)
        if len(self.children) > 0:
            line.append(self.operator)
        else:
            line.append(self.fact_type)

        if self.deduction_type:
            line.append(self.deduction_type)

        if len(self.constraints) > 0:
            line.append(cnsts)

        if len(self.children) > 0:
            line.append(f'자식 노드 개수: {len(self.children)}')

        return ' | '.join(line)

    def __repr__(self):
        return str(self)

    def to_json(self):
        return {
            'value': self.value,
            'children': [x.to_json() for x in self.children],
            'fact_type': self.fact_type,
            'operator': self.operator,
            'constraints': self.constraints,
            'deduction_type': self.deduction_type,
            'prunable': self.prunable,
            'can_be_leaf': self.can_be_leaf
        }

    @classmethod
    def from_json(cls, js):
        js['children'] = [LogicNode.from_json(x) for x in js['children']]
        return cls(**js)


class LogicTree:
    """
    MuSR 예제를 생성할 때 사용되는 주요 데이터 구조입니다. 
    기본적으로 표준 트리 구조이지만, 트리의 형태를 제어하는 여러 매개변수를 포함합니다.
    """

    nodes: List[LogicNode]

    chance_of_or: float
    chance_of_cs_fact: float
    depth: int
    chance_to_prune: float
    chance_to_prune_all: float
    bf_factor: Dict[int, float]
    deduction_type_sample_rate: Dict[LogicNodeDeductionType, float]
    root_structure: List[List[LogicNode]] = ()
    valid: bool = True

    def __init__(
            self,
            chance_of_or: float = 0.3,
            chance_of_cs_fact: float = 0.1,
            depth: int = 2,
            chance_to_prune: float = 0.6,
            chance_to_prune_all: float = 0.2,
            bf_factor: Dict[int, float] = None,
            deduction_type_sample_rate: Dict[LogicNodeDeductionType, float] = None,
            enforce_cs_fact_per_level: bool = False,
            root_structure: List[Any] = (),
            nodes: List[LogicNode] = (),
            populate: bool = True,
            prune: bool = True
    ):
        """
        :param chance_of_or: (사용되지 않음) 자식 노드가 있을 경우 OR가 될 확률.
        :param chance_of_cs_fact: (사용되지 않음) 상식적 사실이 포함될 확률.
        :param depth: 트리의 최대 깊이.
        :param chance_to_prune: 노드를 가지치기할 확률.
        :param chance_to_prune_all: 노드의 모든 자식을 가지치기할 확률.
        :param bf_factor: 분기 계수 (예: {1: 0.33, 2:0.33, 3:0.33}).
        :param deduction_type_sample_rate: (사용되지 않음, bf_factor 및 LogicNodeDeductionType 참조).
        :param enforce_cs_fact_per_level: 각 트리 레벨마다 최소 하나의 상식적 사실을 포함하도록 강제.
        :param root_structure: 논리 트리를 구축할 LogicNode 목록.
        :param nodes: LogicNode 리스트. 값이 설정된 경우 populate 및 prune을 수행하지 않음.
        :param populate: 주어진 매개변수에 따라 자식 노드를 자동 생성할지 여부.
        :param prune: 주어진 매개변수에 따라 가지치기를 수행할지 여부.
        """
        self.chance_of_or = chance_of_or
        self.chance_of_cs_fact = chance_of_cs_fact
        self.depth = depth
        self.chance_to_prune = chance_to_prune
        self.chance_to_prune_all = chance_to_prune_all
        self.bf_factor = bf_factor
        self.enforce_cs_fact_per_level = enforce_cs_fact_per_level
        self.valid = True

        if not bf_factor:
            self.bf_factor = {2: 0.8, 3: 0.2}
        if not deduction_type_sample_rate:
            deduction_type_sample_rate = {LogicNodeDeductionType.SYLLOGISM: 1.0}

        self.deduction_type_sample_rate = deduction_type_sample_rate
        self.root_structure = root_structure

        if len(nodes) > 0:
            self.nodes = nodes
        else:
            if root_structure is not None and len(root_structure) > 0:
                self.nodes = root_structure
            else:
                self.nodes = [LogicNode('루트', operator=LogicNodeOperatorType.AND)]

            if populate:
                [self.populate(x, 1) for x in self.nodes]
            if prune:
                [self.prune(x, 1) for x in self.nodes]

    def __str__(self):
        return self.print_tree()

    def get_facts(self, include_cs: bool = False, include_deductions_from_level: int = -1, no_facts_after_depth: int = -1):
        """
        트리에서 LogicNode 목록을 가져옵니다. 기본적으로 명시적 리프 노드를 반환합니다.

        :param include_cs: 모든 레벨에서 상식적 사실을 포함할지 여부.
        :param include_deductions_from_level: 특정 레벨 이후의 중간 추론 노드를 포함할지 여부.
        :param no_facts_after_depth: 특정 깊이 이후의 노드를 리프 노드로 취급할지 여부.
        """

        def recurse_facts(_node: LogicNode, depth: int = 0) -> List[str]:
            node = deepcopy(_node)
            if depth >= no_facts_after_depth and no_facts_after_depth > -1:
                node.children = []

            facts = []

            if node.fact_type == LogicNodeFactType.EXPLICIT and len(node.children) == 0:
                facts.append(node)
            if node.fact_type == LogicNodeFactType.COMMONSENSE and include_cs and len(node.children) == 0:
                facts.append(node)
            if len(node.children) > 0 and include_deductions_from_level <= depth and include_deductions_from_level > -1:
                facts.append(node)

            for child in node.children:
                facts.extend(recurse_facts(child, depth+1))
            return list(set(facts))

        facts = []
        for n in self.nodes:
            facts.extend(recurse_facts(n))
        return facts

    def print_tree(self, node=None, level=0):
        """ 
        더 이상 사용되지 않는 함수. 
        """
        if node is None:
            node = self.nodes[0]
        line = '-' * level * 4 + str(node) + (" | " + str(node.operator) if len(node.children) > 0 else '')

        for child in node.children:
            line += '\n' + self.print_tree(child, level + 1)

        return line
    def print_for_gpt(
            self,
            node=None,
            level=0,
            pad_char=' ',
            pad_space=4,
            print_forward=True,
            print_conjection_types: bool = False,
            print_reasoning_types: bool = False,
            ignore_value_after_depth: int = -1,
            print_only_nodes_with_value: bool = False
    ):
        """
        Complex print function.  We often use it as
        print_for_gpt(pad_space=1, pad_char='> ')

        However, more complex arguments can be used to control what is printed.

        This returns a string that must be printed (don't be confused by the method name.)

        :param node: Start at a specific node.
        :param level: Controls how much tabbing is done when printing the current node.
        :param pad_char: Char to use that specifies depth ('> ' at depth 3 will look like '> > > ' if you have pad_space equal to 1 for example)
        :param pad_space: How many spaces to include between pad_chars
        :param print_forward: Print the tree with parent nodes first.
        :param print_conjection_types: Print the Ands and Ors per deduction (not used)
        :param print_reasoning_types: Print the deduction types (not used)
        :param ignore_value_after_depth: Ignore content of the nodes once a depth is met
        :param print_only_nodes_with_value: Ignore nodes without content.
        """

        line = ''

        if node is None:
            node = self.nodes[0]

        if not print_forward:
            for child in node.children:
                v = self.print_for_gpt(child, level + 1, pad_char=pad_char, pad_space=pad_space, print_forward=print_forward, ignore_value_after_depth=ignore_value_after_depth, print_only_nodes_with_value=print_only_nodes_with_value)
                if v != '':
                    line += v + '\n'

        ignore_val = ignore_value_after_depth > -1 and ignore_value_after_depth < level
        ignore_line = print_only_nodes_with_value and node.value == ''

        if ignore_line:
            line_val = ''
        else:
            line_val = (node.value + ' | ' if node.value != '' and not ignore_val else '') + (
                (NodeTypeConstants.FACT_FROM_STORY if node.fact_type == LogicNodeFactType.EXPLICIT else NodeTypeConstants.COMMONSENSE) \
                    if len(node.children) == 0 else NodeTypeConstants.DEDUCED_FACT)

            if level == 0:
                line_val = (node.value + ' | ' if node.value != '' else '') + NodeTypeConstants.DEDUCED_ROOT_CONCLUSION 

            if len(node.children) > 0 and (print_conjection_types or print_reasoning_types):
                if print_conjection_types:
                    line_val += f' ({node.operator}'
                else:
                    line_val += f'('
                if node.deduction_type and print_reasoning_types:
                    line_val += f' | {node.deduction_type})'
                else:
                    line_val += ')'

            if len(node.constraints) > 0:
                cnsts = ", ".join([str(x) for x in node.constraints])
                line_val += f' {NodeTypeConstants.CONSTRAINT}: [{cnsts}]'

            line += pad_char * level * pad_space + line_val

        if print_forward:
            for child in node.children:
                v = self.print_for_gpt(child, level + 1, pad_char=pad_char, pad_space=pad_space, print_forward=print_forward, ignore_value_after_depth=ignore_value_after_depth, print_only_nodes_with_value=print_only_nodes_with_value)
                if v != '':
                    line += '\n' + v

        return line

    def populate(self, node: LogicNode, current_depth: int = 1):
        if node.operator == LogicNodeOperatorType.CHOOSE:
            node.operator = LogicNodeOperatorType.OR \
                if random.random() < self.chance_of_or else LogicNodeOperatorType.AND
        if node.deduction_type == LogicNodeDeductionType.CHOOSE:
            if node.operator != LogicNodeOperatorType.AND:
                node.deduction_type = None
            else:
                node.deduction_type = random.choices(list(self.deduction_type_sample_rate.keys()), list(self.deduction_type_sample_rate.values()), k=1)[0]

        if not node.frozen:

            bf = max(0, random.choices(list(self.bf_factor.keys()), list(self.bf_factor.values()), k=1)[0] - len(node.children))

            if bf > 0:


                new_nodes = []
                one_fact_is_cs = False
                for idx in range(bf):
                    roll_for_or = random.random()
                    fact_type = LogicNodeFactType.COMMONSENSE \
                        if random.random() < self.chance_of_cs_fact and not one_fact_is_cs else \
                        LogicNodeFactType.EXPLICIT

                    if roll_for_or > self.chance_of_or and\
                            current_depth < self.depth and\
                            not fact_type == LogicNodeFactType.COMMONSENSE:
                        new_nodes.append(
                            LogicNode(
                                f'',
                                operator=LogicNodeOperatorType.AND,
                                fact_type=fact_type,
                                deduction_type=random.choices(list(self.deduction_type_sample_rate.keys()), list(self.deduction_type_sample_rate.values()), k=1)[0],
                                prunable=True,
                                can_be_leaf=True,
                            )
                        )
                    else:
                        new_nodes.append(
                            LogicNode(
                                f'',
                                operator=LogicNodeOperatorType.OR,
                                fact_type=fact_type,
                                prunable=True,
                                can_be_leaf=True
                            )
                        )

                    if fact_type == LogicNodeFactType.COMMONSENSE:
                        node.operator = LogicNodeOperatorType.AND
                        if not node.deduction_type:
                            node.deduction_type = random.choices(list(self.deduction_type_sample_rate.keys()), list(self.deduction_type_sample_rate.values()), k=1)[0]
                        one_fact_is_cs = True

                if not one_fact_is_cs and self.enforce_cs_fact_per_level:
                    new_nodes.append(LogicNode(f'', operator=LogicNodeOperatorType.OR, fact_type=LogicNodeFactType.COMMONSENSE, prunable=False, can_be_leaf=True))

                node.children.extend(new_nodes)


        if current_depth < self.depth:
            for node in node.children:
                if node.fact_type == LogicNodeFactType.COMMONSENSE:
                    continue
                self.populate(node, current_depth+1)

    def prune(self, node: LogicNode, current_depth: int = 1):
        to_prune = []

        if current_depth > 1 and node.can_be_leaf:
            if random.random() < self.chance_to_prune_all:
                node.children = []
                return

        prunable = [x for x in node.children if x.prunable]
        if (len(prunable) > 1 and node.operator == LogicNodeOperatorType.OR or\
                len(prunable) > 2 and node.operator == LogicNodeOperatorType.AND) and\
                current_depth <= self.depth:

            if node.prunable:
                for n in random.sample(prunable, len(prunable) - (1 if node.operator == LogicNodeOperatorType.OR else 2)):
                    roll_to_prune = random.random()
                    if roll_to_prune < self.chance_to_prune:
                        to_prune.append(n)

        node.children = [x for x in node.children if x not in to_prune]
        for n in node.children:
            self.prune(n, current_depth+1)


    def to_json(self):
        """ 트리를 JSON 형식으로 변환합니다. """
        args = {
            'chance_of_or': self.chance_of_or,
            'depth': self.depth,
            'chance_to_prune': self.chance_to_prune,
            'chance_to_prune_all': self.chance_to_prune_all,
            'bf_factor': self.bf_factor,
            'deduction_type_sample_rate': self.deduction_type_sample_rate,
            'root_structure': [x.to_json() for x in self.root_structure],
            'nodes': [x.to_json() for x in self.nodes]
        }
        return args

    @classmethod
    def from_json(cls, _js):
        """ JSON 데이터를 이용하여 LogicTree를 생성합니다. """
        js = deepcopy(_js)
        js['nodes'] = [LogicNode.from_json(x) for x in js['nodes']]
        js['root_structure'] = [LogicNode.from_json(x) for x in js['root_structure']]
        return cls(**js)



if __name__ == "__main__":
    """ 예제 사용법 """

    def tv_scene_ex():
        root_structure = [
            LogicNode('좋은 드라마 TV 장면', operator=LogicNodeOperatorType.OR,
                      prunable=False, can_be_leaf=False, frozen=True)
        ]

        root_structure[0].children = [
            LogicNode('종민이는 슬프다.', operator=LogicNodeOperatorType.CHOOSE,
                      prunable=True, can_be_leaf=False),
            LogicNode('종민이는 이제 수지를 싫어한다.', operator=LogicNodeOperatorType.CHOOSE,
                      prunable=True, can_be_leaf=False),
            LogicNode('종민이는 차를 샀다.', operator=LogicNodeOperatorType.CHOOSE,
                      prunable=True, can_be_leaf=False),
            LogicNode('종민이는 행복해지고 싶었다.', operator=LogicNodeOperatorType.CHOOSE,
                      prunable=True, can_be_leaf=False),
        ]

        tree = LogicTree(
            depth=4,
            root_structure=root_structure,
            bf_factor={1: 0.5, 2: 0.5},
            chance_of_or=0.0,
            chance_of_cs_fact=0.0,
            chance_to_prune_all=0.5,
            chance_to_prune=0.5,
            enforce_cs_fact_per_level=True
        )

        rep = tree.print_for_gpt(pad_space=1, pad_char='- ')
        print(rep)


    def eb_ex():
        root_structure = [
            LogicNode('', operator=LogicNodeOperatorType.CHOOSE,
                      prunable=False, can_be_leaf=False)
        ]

        n = LogicNode('화산 폭발이 햇빛을 차단한다.', operator=LogicNodeOperatorType.CHOOSE,
                      prunable=False, can_be_leaf=False, frozen=True)
        n.children = [
            LogicNode('화산 폭발이 화산재 구름을 생성한다.', operator=LogicNodeOperatorType.CHOOSE,
                      prunable=False, can_be_leaf=True, frozen=True),
            LogicNode('화산재가 햇빛을 차단한다.', operator=LogicNodeOperatorType.CHOOSE,
                      prunable=False, can_be_leaf=True, frozen=True),
        ]

        g = LogicNode('화산 폭발은 식물을 죽게 할 수 있다.', operator=LogicNodeOperatorType.CHOOSE,
                      prunable=True, can_be_leaf=False, frozen=True)

        g.children = [
            n,
            LogicNode('햇빛 없이는 생산자는 죽는다.', operator=LogicNodeOperatorType.CHOOSE,
                      prunable=False, can_be_leaf=True, frozen=True)
        ]

        l = LogicNode('', operator=LogicNodeOperatorType.AND, prunable=False, can_be_leaf=False)
        l.children = [g]

        root_structure[0].children = [
            l
        ]

        tree = LogicTree(
            depth=5,
            root_structure=root_structure,
            bf_factor={1: 0.3, 2: 0.7},
            chance_of_or=0.0,
            chance_of_cs_fact=0.0,
            chance_to_prune_all=0.0,
            chance_to_prune=0.0,
            enforce_cs_fact_per_level=True
        )

        rep = tree.print_for_gpt(pad_space=1, pad_char='- ')
        print(rep)


    def murder_mystery_ex():
        root_structure = [
            LogicNode('살인자', operator=LogicNodeOperatorType.OR,
                      constraints=[LogicNodeConstraints.ONLY_ONE_CAN_BE_TRUE], prunable=False, can_be_leaf=False,
                      frozen=True)
        ]

        suspect_nodes = [LogicNode(f'용의자 {idx + 1}', operator=LogicNodeOperatorType.AND, prunable=False,
                                   can_be_leaf=False, frozen=True) for idx in range(1)]
        for s in suspect_nodes:
            s.children = [
                LogicNode('용의자가 수단을 가지고 있다.', operator=LogicNodeOperatorType.CHOOSE, prunable=True, can_be_leaf=False),
                LogicNode('용의자가 동기를 가지고 있다.', operator=LogicNodeOperatorType.CHOOSE, prunable=True,
                          can_be_leaf=False),
                LogicNode('용의자가 기회를 가지고 있다.', operator=LogicNodeOperatorType.CHOOSE, prunable=True,
                          can_be_leaf=False)
            ]
        root_structure[0].children = suspect_nodes

        tree = LogicTree(
            depth=4,
            root_structure=root_structure,
            bf_factor={1: 0.5, 2: 0.5},
            chance_of_or=0.0,
            chance_of_cs_fact=0.0,
            chance_to_prune_all=0.5,
            chance_to_prune=0.5,
            enforce_cs_fact_per_level=True
        )

        rep = tree.print_for_gpt(pad_space=1, pad_char='> ')
        print(rep)


    def action_ex():
        root_structure = [
            LogicNode('행동을 취하다', operator=LogicNodeOperatorType.OR, prunable=False, can_be_leaf=False,
                      frozen=True)
        ]

        root_structure[0].children = [
            LogicNode('도망간다', operator=LogicNodeOperatorType.CHOOSE, prunable=False, can_be_leaf=False,
                      frozen=True),
            LogicNode('맞서 싸운다', operator=LogicNodeOperatorType.CHOOSE, prunable=False, can_be_leaf=False,
                      frozen=True),
            LogicNode('숨는다', operator=LogicNodeOperatorType.CHOOSE, prunable=False, can_be_leaf=False, frozen=True),
        ]

        for cidx, c in enumerate(root_structure[0].children):
            nfacts = random.randint(2, 4)

            for n in range(nfacts):
                fact = LogicNode('', operator=LogicNodeOperatorType.CHOOSE, prunable=False, can_be_leaf=False,
                                 frozen=True)
                fact.children = [
                    LogicNode('장점 (부모 행동을 지지하는 내용)', operator=LogicNodeOperatorType.CHOOSE,
                              prunable=True, can_be_leaf=False,
                              frozen=False),
                    LogicNode('단점 (형제 장점에 반대하는 내용)', operator=LogicNodeOperatorType.CHOOSE,
                              prunable=True, can_be_leaf=False, frozen=False)
                ]
                root_structure[0].children[cidx].children.append(fact)

        tree = LogicTree(
            depth=4,
            root_structure=root_structure,
            bf_factor={1: 0.25, 2: 0.5, 3: 0.25},
            chance_of_or=0.0,
            chance_of_cs_fact=0.0,
            chance_to_prune_all=0.5,
            chance_to_prune=0.75,
            enforce_cs_fact_per_level=True
        )

        rep = tree.print_for_gpt(pad_space=1, pad_char='- ')
        print(rep)


    tv_scene_ex()
    eb_ex()
    action_ex()
