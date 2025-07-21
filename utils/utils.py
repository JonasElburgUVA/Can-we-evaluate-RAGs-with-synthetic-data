"""
General helper functions to keep notebooks clean
"""
import os
import json
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from ragas.llms import LangchainLLMWrapper


per_model_costs={
    "gpt-4o-mini-2024-07-18" : [0.15e-6, 0.6e-6],
    "gpt-4o-2024-08-06" : [2.5e-6, 10e-6],
    "llama" : [0.07e-6, 0.2e-6],
    "gpt-3.5-turbo" : [0.5e-6, 1.5e-6],
    "haiku": [0.25e-6, 1.25e-6],
} # cost per [input, output] token

def get_evaluator_models(llm_deployment :str, embedding_deployment:str):
    endpoint = os.getenv("ENDPOINT_URL")
    token_provider = get_bearer_token_provider(  
        DefaultAzureCredential(),  
        "https://cognitiveservices.azure.com/.default"  
    )  

    evaluator_llm = LangchainLLMWrapper(
        AzureChatOpenAI(
            api_version="2024-05-01-preview",
            azure_endpoint=endpoint,  
            azure_ad_token_provider=token_provider,  
            azure_deployment=llm_deployment,
        )
    )

    embedding_model = LangchainEmbeddingsWrapper(
        AzureOpenAIEmbeddings(
            api_version="2024-05-01-preview",
            azure_ad_token_provider=token_provider,  
            azure_endpoint=endpoint,
            azure_deployment=embedding_deployment
        )
    )

    return evaluator_llm, embedding_model


