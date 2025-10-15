STORY = "이야기"


class NodeTypeConstants:
    COMPLEX_FACT = "복합 사실"
    FACT_FROM_STORY = "스토리에서 추출된 사실"
    COMMONSENSE_KNOWLEDGE = "상식"
    DEDUCED_CONCLUSION = "추론된 결론"
    DEDUCED_ROOT_CONCLUSION = "추론된 결론"
    DEDUCED_FACT = "추론된 사실"

    EXPLICIT = "사실"
    COMMONSENSE = "상식"

    CONSTRAINT = "제약"
    

class Conjunctions:
    BECAUSE = "왜냐하면"
    AND = "그리고"
    OR = "또는"
    UNLESS = "상황에 따라 달라질 수 있다"
    BESIDES = "한편"

class PromptStep:
    SCENARIO = "시나리오:"
    CURRENT_TREE = "현재 트리:"
    ENTAILMENT_STEP_TO_COMPLETE = "완성해야 할 함의 추론:"
    ENTAILMENT_STEP_TO_COMPLETE_STR = "완성해야 할 함의 추론"

    @classmethod
    def remove_colon(cls, step):
        return step[:-1] if step.endswith(":") else step

class MysteryConstants:
    VICTIM = "피해자"
    PLACE = "범행 장소"
    TOOL = "살해 도구"
    SUSPECT = "용의자"
    ROLE = "용의자의 역할"
    MOTIVE = "범행 동기"
    RED_HERRINGS = "허위 단서"
    MEANS = "범행 수단"
    OPPORTUNITY = "범행 기회"
    MURDERER = "살인자"

    @staticmethod
    def determine_josa(name):
        """
        이름의 받침 여부에 따라 적절한 조사를 반환하는 메서드.
        """
        last_char = name[-1]
        unicode_val = ord(last_char)
        if 44032 <= unicode_val <= 55203:
            # 한글 음절이면 받침 여부 판단
            if (unicode_val - 44032) % 28 == 0:
                return "는"  # 받침 없음
            else:
                return "이는"  # 받침 있음
        else:
            return "는"  # 한글이 아니거나 특수 문자

    @staticmethod
    def determine_is(name):
        # 가수다, 경찰이다
        last_char = name[-1]
        unicode_val = ord(last_char)
        if 44032 <= unicode_val <= 55203:
            # 한글 음절이면 받침 여부 판단
            if (unicode_val - 44032) % 28 == 0:
                return "다"  # 받침 없음
            else:
                return "이다"  # 받침 있음
        else:
            return "이다"  # 한글이 아니거나 특수 문자

    @classmethod
    def is_murderer(cls, character):
        josa = cls.determine_josa(character)
        return f"{character}{josa} {cls.MURDERER}다."

    @classmethod
    def has_means(cls, character):
        josa = cls.determine_josa(character)
        return f"{character}{josa} {cls.MEANS}이 있다."

    @classmethod
    def has_motive(cls, character):
        josa = cls.determine_josa(character)
        return f"{character}{josa} {cls.MOTIVE}가 있다."
    
    @classmethod
    def has_opportunity(cls, character):
        josa = cls.determine_josa(character)
        return f"{character}{josa} {cls.OPPORTUNITY}가 있다."


class ObjectPlacementConstants:
    MOVER = "이동자"
    OBJECT = "물건"
    SRC = "출발 위치"
    DEST = "도착 위치"
    REASON = "이유"
    NAME = "이름"
    ROLE = "역할"
    MOTIVE = "동기"
    WHEN_MOVING = "이동할 때"

    @classmethod
    def determine_josa(cls, name):
        """
        이름의 받침 여부에 따라 적절한 조사를 반환하는 메서드.
        """
        last_char = name[-1]
        unicode_val = ord(last_char)
        if 44032 <= unicode_val <= 55203:
            # 한글 음절이면 받침 여부 판단
            if (unicode_val - 44032) % 28 == 0:
                return "는"  # 받침 없음
            else:
                return "이는"  # 받침 있음
        else:
            return "는"  # 한글이 아니거나 특수 문자
    @classmethod
    def determine_josa_for_object(cls, obj):
        last_char = obj[-1]
        unicode_val = ord(last_char)
        if 44032 <= unicode_val <= 55203:
            # 한글 음절이면 받침 여부 판단
            if (unicode_val - 44032) % 28 == 0:
                return "를"
            else:
                return "을"
        else:
            return "를"
    @classmethod
    def determine_josa_for_dest(cls, dest):
        # 약수터로, 집으로
        last_char = dest[-1]
        unicode_val = ord(last_char)
        if 44032 <= unicode_val <= 55203:
            # 한글 음절이면 받침 여부 판단
            if (unicode_val - 44032) % 28 == 0:
                return "로"
            else:
                return "으로"
        else:
            return "로"
    @classmethod
    def determine_josa_for_mover(cls, name):
        # 현우가, 지민이가
        last_char = name[-1]
        unicode_val = ord(last_char)
        if 44032 <= unicode_val <= 55203:
            # 한글 음절이면 받침 여부 판단
            if (unicode_val - 44032) % 28 == 0:
                return "가"
            else:
                return "이"
        else:
            return "가"
    @classmethod
    def determine_josa_for_subject(cls, name):
        # 노트북은, 공책은, 집게는
        last_char = name[-1]
        unicode_val = ord(last_char)
        if 44032 <= unicode_val <= 55203:
            # 한글 음절이면 받침 여부 판단
            if (unicode_val - 44032) % 28 == 0:
                return "는"
            else:
                return "은"
        else:
            return "은"
   
    @classmethod
    def add_josa(cls, objet, mode):
        objet = objet.strip()
        if mode == "object":
            return f"{objet}{cls.determine_josa_for_object(objet)}"
        elif mode == "dest":
            return f"{objet}{cls.determine_josa_for_dest(objet)}"
        elif mode == "mover":
            return f"{objet}{cls.determine_josa_for_mover(objet)}"
        elif mode == "name":
            return f"{objet}{cls.determine_josa(objet)}"
        elif mode == "subject":
            return f"{objet}{cls.determine_josa_for_subject(objet)}"
        else:
            return objet
     
        
    @classmethod
    def move(cls, name, objet, dest):
        # note that the variable name is objet, not a reserved name 'object'
        msg = f"{cls.add_josa(name, 'mover')} {cls.add_josa(objet, 'object')} {cls.add_josa(dest, 'dest')} 옮긴다."
        # print("MOVE:", msg)
        return msg


class TeamAllocationConstants:
    PEOPLE = "사람들"
    ROLE = "역할"
    CAPABILITIES = "능력"

    
    @classmethod
    def determine_josa(cls, name):
        """
        이름의 받침 여부에 따라 적절한 조사를 반환하는 메서드.
        """
        last_char = name[-1]
        unicode_val = ord(last_char)
        if 44032 <= unicode_val <= 55203:
            # 한글 음절이면 받침 여부 판단
            if (unicode_val - 44032) % 28 == 0:
                return "는"  # 받침 없음
            else:
                return "이는"  # 받침 있음
        else:
            return "는"  # 한글이 아니거나 특수 문자
    @classmethod
    def determine_josa_for_object(cls, obj):
        last_char = obj[-1]
        unicode_val = ord(last_char)
        if 44032 <= unicode_val <= 55203:
            # 한글 음절이면 받침 여부 판단
            if (unicode_val - 44032) % 28 == 0:
                return "를"
            else:
                return "을"
        else:
            return "를"
    @classmethod
    def determine_josa_for_dest(cls, dest):
        # 약수터로, 집으로
        last_char = dest[-1]
        unicode_val = ord(last_char)
        if 44032 <= unicode_val <= 55203:
            # 한글 음절이면 받침 여부 판단
            if (unicode_val - 44032) % 28 == 0:
                return "로"
            else:
                return "으로"
        else:
            return "로"
    @classmethod
    def determine_josa_for_mover(cls, name):
        # 현우가, 지민이가
        last_char = name[-1]
        unicode_val = ord(last_char)
        if 44032 <= unicode_val <= 55203:
            # 한글 음절이면 받침 여부 판단
            if (unicode_val - 44032) % 28 == 0:
                return "가"
            else:
                return "이"
        else:
            return "가"
    @classmethod
    def determine_josa_for_subject(cls, name):
        # 노트북은, 공책은, 집게는
        last_char = name[-1]
        unicode_val = ord(last_char)
        if 44032 <= unicode_val <= 55203:
            # 한글 음절이면 받침 여부 판단
            if (unicode_val - 44032) % 28 == 0:
                return "는"
            else:
                return "은"
        else:
            return "은"
    
    @classmethod
    def determine_josa_for_conjuction(cls, name):
        # 지수와, 규성과
        last_char = name[-1]
        unicode_val = ord(last_char)
        if 44032 <= unicode_val <= 55203:
            # 한글 음절이면 받침 여부 판단
            if (unicode_val - 44032) % 28 == 0:
                return "와"
            else:
                return "과"
        else:
            return "과"
   
    @classmethod
    def add_josa(cls, objet, mode):
        objet = objet.strip()
        if mode == "object":
            return f"{objet}{cls.determine_josa_for_object(objet)}"
        elif mode == "dest":
            return f"{objet}{cls.determine_josa_for_dest(objet)}"
        elif mode == "mover":
            return f"{objet}{cls.determine_josa_for_mover(objet)}"
        elif mode == "name":
            return f"{objet}{cls.determine_josa(objet)}"
        elif mode == "subject":
            return f"{objet}{cls.determine_josa_for_subject(objet)}"
        elif mode == "conjunction":
            return f"{objet}{cls.determine_josa_for_conjuction(objet)}"
        else:
            return objet
     

