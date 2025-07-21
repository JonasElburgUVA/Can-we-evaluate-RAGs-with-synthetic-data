from enum import Enum

class DatasetType(str, Enum):
    SQUAD = "SQuAD"
    PEGA_QUESTIONS = "pega_questions"
    LAUNCHPAD = "Launchpad"
    ASQA = "ASQA"
    SALES = "Sales"

class MetricGroup(str, Enum):
    NON_LLM = "non_llm"
    SUPERVISED = "supervised"
    UNSUPERVISED = "unsupervised"
    FULL = "full"
    CUSTOM = "custom"