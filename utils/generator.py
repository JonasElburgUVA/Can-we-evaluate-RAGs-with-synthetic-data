import os
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import tqdm

class Generator():
    def __init__(self, 
                 prompt : ChatPromptTemplate, 
                 prompt_keys : list, 
                 **kwargs):
        endpoint = os.getenv("ENDPOINT_URL")
        token_provider = get_bearer_token_provider(
            DefaultAzureCredential(),
            "https://cognitiveservices.azure.com/.default"  
        )
        self.kwargs = kwargs
        self.temperature = kwargs.get("temperature", 0)
        self.seed = kwargs.get("seed", 0)

        self.generator = AzureChatOpenAI(
            api_version="2024-05-01-preview",
            azure_endpoint=endpoint,
            azure_ad_token_provider=token_provider,
            temperature=self.temperature,
            seed=self.seed,
            azure_deployment="gpt-4o-mini"
            )
        
        self.prompt = prompt
        self.prompt_keys = prompt_keys
        self.chain = self.prompt | self.generator
        assert 1==1


        
        
    
