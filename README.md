# RAG-Evaluation-with-Minimal-Labels
Repository containing the code and documentation for the 3-month internship with Pega concerning evaluation of retrieval augmented generative (RAG) models, internally also known as knowledge buddies.This repository enables the evaluation of knowledge buddies using synthetic or your own evaluation dataset of questions with reference answers. 

## Structure of code
The primary purpose of this research was to evaluate the reliability of a synthetic question-answering (QA) benchmark compared to a human-curated benchmark. The experiments are structured around the following four notebooks:

1. Preprocessing

The preprocessing step serves to format the corpus and human-curated benchmark in easy to use json files. For instructions on the preprocessing of your own data, refer to [this notebook](https://github.com/JonasElburgUVA/Can-we-evaluate-RAGs-with-synthetic-data/blob/main/Experiments/preprocessing/pp_template.ipynb).

2. Ingestion

In the ingestion step, the corpus is uploaded, chunked, and tokenized, after which it is accessible on the Knowledge Buddy platform under the specified collection and dataSource. We use a simple ingestion pipeline without additional attributes, where only the chunking parameters are adjustable by changing the template file. The ingestion notebook should be extended if you wish to add attributes.

3. Inference

After ingesting data, you specify your experiment arguments in a [config](#config) file. After having specified identifiers, buddies, deployments, and evaluation metrics in this file, the inference notebook carries out the main part of the experiments. First, we make predictions on the human-curated benchmark using the specified set of buddies. Then we do the same for the synthetic benchmark. Evaluation metrics are calculated for all buddy-benchmark combinations.

4. Analysis

In this step we analyse the results, focussing on the buddy ranking consistency between human and synthetic benchmarks.

## Configurating the experiment
After preprocessing, the ingestion, inference, and analysis can be reproduced using only a configuration file. Config files are saved using yaml under `configs/experiments/`. The available arguments are the following:

```yaml
# General
experiment_name:               # (string) A custom name for your experiment. Useful to avoid overwriting previous runs.
dataset_name:                  # (string) Folder name where the dataset is saved.

# Buddies
buddies:                       # (list) List of model names (buddies) used in the experiment.
  - model1
  - model2
baseline:                      # (string) One of the buddies. Used in visualizations as a baseline reference.

# Synthetic Data Generation
generator_llm_deployment:      # (string) Azure OpenAI deployment name used for generating synthetic Q&A pairs.
num_source_docs:               # (int) Number of source documents used for generation.
num_gen_questions_per_document: # (int) Number of questions to generate per document.
generated_dataset_name:        # (string) Identifier for the generated dataset.

# Evaluation
eval_llm_deployment:           # (string) Azure OpenAI deployment used for LLM-based metrics (e.g. with RAGAS).
eval_embedding_deployment:     # (string) Azure OpenAI deployment used for embedding-based metrics (e.g. with RAGAS).
metric_group:                  # (string) Metric preset group. One of:
                               #          - supervised (BLEU, ROUGE-L, Levenshtein Similarity, Semantic Similarity). Requres embeddings.
                               #          - full (supervised metrics + Answer Relevancy, Context Precision, Faithfulness). Requires embeddings + LLM.
                               #          - non-llm (BLEU, ROUGE-L, Levenshtein Similarity).
                               #          - custom (specify metrics yourself)

# Optional Settings
use_doc_ids:                   # (bool) Set to true if document IDs are known for the subset of documents you want to select from the human question benchmarks. Only used in SQuAD.
min_source_doc_length:         # (int) minimum length of documents used to generate synthetic question-answer pairs. In report only used for Sales (=500), but likely a useful feature to default set to a few hundred.
num_human_questions:           # (int) number of questions in the human evaluation benchmark. The load_human_questions method normally attempts to load the same amount of questions as are in the synthetic dataset. You can use this parameter if you have a different number of questions in your human benchmark.

```

## Running a single evaluation
While the research focussed on comparing human and synthetic benchmarks, the code also enables evaluation using only one benchmark, synthetic or human. For an example, see the [single evaluation example](https://github.com/JonasElburgUVA/Can-we-evaluate-RAGs-with-synthetic-data/blob/main/Experiments/example.ipynb) notebook.

## Environment variables
The following environment variables are *required* for the code to work:

- PROJECT_ROOT - Your root directory
- PEGA_BASE_URL - the Pega KnowledgeBuddy endpoint
- PEGA_USERNAME - Pega username used on the knowledge buddy platform
- PEGA_PASSWORD - Matching password
- AZURE_OPENAI_ENDPOINT - the oai endpoint

The following environmnt variables are optional:

- RAGAS_DO_NOT_TRACK (optional) - To opt-out of RAGAS usage data collection, set this environment variable to true. 
- RAGAS_APP_TOKEN - if you want to upload results to the RAGAS app to analyse the evaluation results, set your app token here.

