from typing import TypedDict, Annotated
from langgraph.graph import StateGraph , END
import operator


class AgentState(TypedDict):
    query: str
    documents : list[dict]
    answer: str
    web_search_needed: bool
    grade: str
    num_retrieves: str
    