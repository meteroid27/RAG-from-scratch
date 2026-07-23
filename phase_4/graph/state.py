from typing import TypedDict


class AgentState(TypedDict):
    query: str
    documents : list[dict]
    answer: str
    'web_search_needed: bool'
    grade: str
    retrieval_path : str
    num_retries : int 
    cache_hit: bool