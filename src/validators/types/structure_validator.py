from typing import List

from src.logic_tree.tree import LogicNode, LogicNodeFactType
from src.validators.validator import Validator


class StructureValidator(Validator):
    """
    간단한 검증기로, 템플릿 노드의 명시적 사실과 상식적 사실의 개수가 
    생성된 것과 동일한지 확인합니다.
    """

    def validate(
            self,
            template: LogicNode,
            explicit_facts: List[str],
            commonsense_facts: List[str],
            raw_output: str,
            *args,
            **kwargs
    ) -> bool:
        """
        템플릿 노드와 생성된 사실의 개수를 비교하여 검증합니다.

        :param template: 검증할 LogicNode 템플릿.
        :param explicit_facts: 생성된 명시적 사실 목록.
        :param commonsense_facts: 생성된 상식적 사실 목록.
        :param raw_output: 원본 출력 문자열.
        :return: 구조가 일치하면 True, 그렇지 않으면 False.
        """
        return \
            len(explicit_facts) == len([
                x for x in template.children if x.fact_type == LogicNodeFactType.EXPLICIT
            ]) and \
            len(commonsense_facts) == len([
                x for x in template.children if x.fact_type == LogicNodeFactType.COMMONSENSE
            ])

    def retry_prompt(
            self,
            template: LogicNode,
            explicit_facts: List[str],
            commonsense_facts: List[str],
            raw_output: str,
            *args,
            **kwargs
    ) -> str:
        """
        검증 실패 시, 다시 생성하도록 요청하는 프롬프트를 반환합니다.

        :param template: 현재 검증 중인 LogicNode.
        :param explicit_facts: 생성된 명시적 사실 목록.
        :param commonsense_facts: 생성된 상식적 사실 목록.
        :param raw_output: 원본 추론 출력.
        :return: 재생성을 요청하는 문자열.
        """
        return f'''
이전 출력:

{raw_output}

이전 결과는 구조가 일치하지 않거나 동일한 사실을 두 번 포함하고 있습니다.  

이번에는 정확한 개수의 "스토리에서 추출된 사실(Facts From Story)"과 "상식적 지식(Commonsense Knowledge)"을 포함하고, 중복되지 않도록 생성하세요.

        '''.strip()
