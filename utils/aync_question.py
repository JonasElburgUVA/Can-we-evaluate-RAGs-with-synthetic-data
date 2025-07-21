from pdstools import Infinity
import httpx
from typing import Optional, List
from functools import partial
from pdstools.infinity.resources.knowledge_buddy.knowledge_buddy import (
    TextInput,
    FilterAttributes,
    BuddyResponse,
)
import os

client = Infinity.from_basic_auth(
    os.environ.get("PEGA_BASE_URL"),
    os.environ.get("PEGA_USERNAME"),
    os.environ.get("PEGA_PASSWORD"),
    pega_version="24.2",
    timeout=90,
)
async def question_async(
    self,
    question: str,
    buddy: str,
    include_search_results: bool = False,
    question_source: Optional[str] = None,
    question_tag: Optional[str] = None,
    additional_text_inputs: Optional[List[TextInput]] = None,
    filter_attributes: Optional[List[FilterAttributes]] = None,
    user_name: Optional[str] = None,
    user_email: Optional[str] = None,
) -> BuddyResponse:
    """Send a question to the Knowledge Buddy.

    Parameters
    ----------
    question: str: (Required)
        Input the question.
    buddy: str (Required)
        Input the buddy name.
        If you do not have the required role to access the buddy,
        an access error will be displayed.
    include_search_results: bool (Default: False)
        If set to true, this property returns chunks of data related to each
        SEARCHRESULTS information variable that is defined for the Knowledge Buddy,
        which is the same information that is returned during a semantic search.
    question_source: str (Optional)
        Input a source for the question based on the use case.
        This information can be used for reporting purposes.
    question_tag: str (Optional)
        Input a tag for the question based on the use case.
        This information can be used for reporting purposes.
    additional_text_inputs: List[TextInput]: (Optional)
        Input the search variable values, where key is the search variable name
        and value is the data that replaces the variable.
        Search variables are defined in the Information section of the Knowledge Buddy.
    filter_attributes: List[FilterAttributes]: (Optional)
        Input the filter attributes to get the filtered chunks from the vector database.
        User-defined attributes ingested with content can be used as filters.
        Filters are recommended to improve the semantic search performance.
        Database indexes can be used further to enhance the search.
    """

    async_client = httpx.AsyncClient(
        auth=self._client.auth,
        base_url=self._client.base_url,
        timeout=self._client.timeout,
    )
    # print(f"Calling question {question}")
    response = await async_client.post(
        self._client.base_url.join("/prweb/api/knowledgebuddy/v1/question"),
        json=dict(
            question=question,
            buddy=buddy,
            includeSearchResults=include_search_results,
            questionSource=question_source,
            questionTag=question_tag,
            additionalTextInputs=additional_text_inputs,
            filterAttributes=filter_attributes,
            userName=user_name,
            userEmail=user_email,
        ),
    )
    # print(f"Succesfully answered question {question}")
    try:
        return BuddyResponse(**response.json())
    except:
        print(f"WARNING: {response.status_code} error raised in question {question}. Is it possible this question triggered content filters or rate limits?")
        return response


client.knowledge_buddy.question_async = partial(question_async, self=client) 