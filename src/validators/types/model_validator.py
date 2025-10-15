import sys
from typing import List, Union, Tuple, Optional

from src.logic_tree.tree import LogicNode, LogicNodeFactType
from src.validators.validator import Validator
from src.model import Model


class ModelValidator(Validator):
    """
    언어 모델을 사용하여 주어진 추론에 대한 질문을 생성하고, 모델의 응답을 확인하는 검증기입니다.

    특정 모델이 낮은 오탐률을 가질 경우, 이를 사용하여 조기 탈출을 수행할 수 있습니다.
    (예: GPT-3.5를 사용하여 조기 탈출)
    """
    def __init__(
            self,
            model: Model,
            prompt: str,
            reason_why: str,
            answer_for_validity: str = 'no',
            conditional: Optional[str] = None,
            early_escape_model: Optional[Model] = None
    ):
        """
        :param model: 검증을 수행할 메인 모델.
        :param prompt: 모델에게 제공할 질문.
        :param reason_why: 검증 실패 시, 재시도를 요청하는 이유.
        :param answer_for_validity: 모델의 응답에서 추론이 유효하다고 간주할 키워드 (예: "no").
        :param conditional: 특정 부모 노드가 포함해야 할 단어. 해당 단어가 없으면 검증을 건너뜀.
        :param early_escape_model: 먼저 실행하여 answer_for_validity 값이 일치하면 메인 모델 호출을 생략할 모델.
        """

        self.model = model
        self.prompt = prompt
        self.reason_why = reason_why
        self.answer_for_validity = answer_for_validity
        self.conditional = conditional
        self.early_escape_model = early_escape_model

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
        주어진 논리 트리가 유효한지 확인합니다.

        :param template: 현재 검증 중인 LogicNode.
        :param explicit_facts: 명시적 사실 목록.
        :param commonsense_facts: 상식적 사실 목록.
        :param raw_output: 원본 추론 출력.
        :return: 유효하면 True, 그렇지 않으면 False.
        """

        if self.conditional:
            check_validity = False

            p = template.parent
            while p is not None:
                if self.conditional.lower() in p.value.lower():
                    check_validity = True
                    break
                p = p.parent

            if not check_validity:
                return True  # 조건이 충족되지 않으면 검증을 건너뜀

        # 조기 탈출 모델 사용
        if self.early_escape_model:
            early_prompt = f'{self.prompt}\n\n추론 내용:\n{raw_output}\n\n판단한 이유를 단계별로 설명하고, 다음 형식으로 답변하세요:\nANSWER: (yes/no)'
            early_output = self.early_escape_model.inference(early_prompt)
            early_output = early_output["choices"][0]['message']['content']
            early_answer = early_output.split('ANSWER:')[-1]

            if self.answer_for_validity.lower() in early_answer.lower():
                return True  # 조기 탈출 조건 충족 시 검증 성공

        # 메인 모델을 사용한 검증
        prompt = f'{self.prompt}\n\n추론 내용:\n{raw_output}\n\n짧은 설명을 작성한 후 다음 형식으로 답변하세요:\nANSWER: (yes/no)'
        output = self.model.inference(prompt)
        output = output["choices"][0]['message']['content']
        answer = output.split('ANSWER:')[-1]

        return self.answer_for_validity.lower() in answer.lower()

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
        :param explicit_facts: 명시적 사실 목록.
        :param commonsense_facts: 상식적 사실 목록.
        :param raw_output: 원본 추론 출력.
        :return: 재생성을 요청하는 문자열.
        """
        return f'''
이전 출력:

{raw_output}

이 결과는 올바르지 않습니다.  

{self.reason_why}
        '''.strip()
