from ragas.metrics import BleuScore, RougeScore, ExactMatch, StringPresence, SemanticSimilarity, LLMContextPrecisionWithoutReference, LLMContextRecall, Faithfulness, ResponseRelevancy
from ragas.metrics._string import NonLLMStringSimilarity
from ragas.metrics._factual_correctness import FactualCorrectness
from functools import partial

def non_llm_scorers(bleu_ueo=True, **kwargs):
    scorers = [
                BleuScore(kwargs={"use_effective_order": bleu_ueo}),
                RougeScore(),
                StringPresence(),
                NonLLMStringSimilarity(),
    ]
    return scorers

def supervised_scorers(bleu_ueo=True, **kwargs):
    scorers = [
                BleuScore(kwargs={"use_effective_order": bleu_ueo}),
                RougeScore(),
                StringPresence(),
                NonLLMStringSimilarity(),
                SemanticSimilarity()
    ]
    return scorers

def unsupervised_scorers(**kwargs):
    scorers = [
        LLMContextPrecisionWithoutReference(),
        # LLMContextRecall(),
        Faithfulness(),
        ResponseRelevancy()
    ]
    return scorers

def full_scorers():
    scorers = supervised_scorers()
    scorers += unsupervised_scorers()
    return scorers

def custom_scorers(selection, **kwargs):
    "Kwargs should include LLM and embeddingmodel when selecting metrics that require these"
    metrics = {
        "BLEU" : partial(BleuScore, use_effective_order=True),
        "ROUGE" : RougeScore,
        "String Presence": StringPresence,
        "Levenshtein String Similarity": NonLLMStringSimilarity,
        "Semantic similarity": SemanticSimilarity,
        # "Factual correctness" : FactualCorrectness,
        # context relevance
        "Context Precision": LLMContextPrecisionWithoutReference,
        # "Context Recall": LLMContextRecall,
        # faithfulness
        "Faithfulness" : Faithfulness,
        # response relevance
        "Response relevance": ResponseRelevancy}
    
    for k in selection:
        if k not in metrics.keys():
            print(f"warning: the metric '{k}' is selected, but is not implemented.")
    scorers = [metrics[k](**kwargs) for k in selection]
    return scorers