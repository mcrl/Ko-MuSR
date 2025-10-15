from typing import List, Union, Tuple

from src.logic_tree.tree import LogicNode, LogicNodeFactType
from src.validators.validator import Validator


class ForbiddenTextValidator(Validator):
    """
    명시적 사실 및 상식적 사실의 텍스트에서 특정 금지된 단어를 검사하는 검증기입니다. 
    해당 단어가 포함될 경우, 추론은 유효하지 않은 것으로 간주됩니다.

    또한, 특정 부모 노드 내용에 따라 금지된 단어를 조건부로 검사할 수도 있습니다.
    예를 들어, "수단" 분기에서 "동기"를 언급하는 추론을 가지치기할 수 있습니다.
    """

    def __init__(
            self,
            forbidden_words: List[Union[Tuple[str, str], str]],
            reason_why: str = None
    ):
        """
        :param forbidden_words: 금지된 단어 목록. 문자열일 경우 해당 단어는 모든 경우에서 금지됩니다.
            리스트 형태일 경우 ["조건부 단어", "금지된 단어"]로 구성되며, 부모 노드에 "조건부 단어"가 포함되어 있을 경우에만 
            새로운 추론에서 "금지된 단어"를 검사합니다.
        :param reason_why: 특정 단어를 금지하는 이유를 설명하는 메시지 (재생성 프롬프트에서 사용됨).
        """

        self.forbidden_words = [x for x in forbidden_words if x != '']
        self.reason_why = reason_why

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
        주어진 추론이 금지된 단어를 포함하는지 확인합니다.

        :param template: 현재 검증 중인 LogicNode.
        :param explicit_facts: 명시적 사실 목록.
        :param commonsense_facts: 상식적 사실 목록.
        :param raw_output: 원본 출력 문자열.
        :return: 금지된 단어가 포함되지 않았다면 True, 포함되었다면 False.
        """

        for word in self.forbidden_words:
            if isinstance(word, str):
                # 단순 금지 단어 검사
                forbidden_word = word
            else:
                # 조건부 단어 검사
                conditional_text = word[0]
                forbidden_word = word[1]

                check_validity = False

                p = template
                while p is not None:
                    if conditional_text.lower() in p.value.lower():
                        check_validity = True
                        break
                    p = p.parent

                if not check_validity:
                    continue

            if any([forbidden_word.lower() in x.lower() for x in explicit_facts]) or \
               any([forbidden_word.lower() in x.lower() for x in commonsense_facts]):
                print("금지된 단어 발견:", forbidden_word)
                return False
        
        return True

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
        금지된 단어가 포함되었을 경우, 다시 생성하도록 요청하는 프롬프트를 생성합니다.

        :param template: 현재 검증 중인 LogicNode.
        :param explicit_facts: 명시적 사실 목록.
        :param commonsense_facts: 상식적 사실 목록.
        :param raw_output: 원본 출력 문자열.
        :return: 재생성을 요청하는 문자열.
        """

        used_forbidden_words = []
        for word in self.forbidden_words:
            if isinstance(word, str):
                forbidden_word = word
            else:
                conditional_text = word[0]
                forbidden_word = word[1]

                check_validity = False

                p = template
                while p is not None:
                    if conditional_text.lower() in p.value.lower():
                        check_validity = True
                        break
                    p = p.parent

                if not check_validity:
                    continue
            if any([forbidden_word.lower() in x.lower() for x in explicit_facts]) or \
                any([forbidden_word.lower() in x.lower() for x in commonsense_facts]):
                used_forbidden_words.append(forbidden_word)
            # used_forbidden_words.append(forbidden_word)
        used_forbidden_words_str = '\n'.join([f'- {x}' for x in used_forbidden_words])
        reason_str = f'\n이 단어들을 피해야 하는 이유: {self.reason_why}' if self.reason_why else ''

        return f'''
        
이전 출력:

{raw_output}

출력된 일부 사실에 포함되지 않아야 할 내용이 포함되어 있습니다. 
다음 단어를 사용하지 않고 다시 생성하세요:

{used_forbidden_words_str}
{reason_str}
        '''.strip()
