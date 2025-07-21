from .metrics import non_llm_scorers, supervised_scorers, unsupervised_scorers, full_scorers, custom_scorers
from .enums import MetricGroup

def get_metric_group(name, **kwargs):
    """
    non_llm
    supervised
    unsupervised
    full
    custom
    """
    metric_group = {
        MetricGroup.NON_LLM : non_llm_scorers,
        MetricGroup.SUPERVISED : supervised_scorers,
        MetricGroup.UNSUPERVISED : unsupervised_scorers,
        MetricGroup.FULL : full_scorers,
        MetricGroup.CUSTOM : custom_scorers
    }[name](**kwargs)

    return metric_group