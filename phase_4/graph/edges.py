from langgraph.graph import END
from state import AgentState

def decide_retrieve_path(state):
    
    if state["web_search_needed"]:
        return "web_search"
    return "rag_retrieve"

def decide_after_grading(state):
    
    if state["grade"] == "good":
        return "cache_result"
    
    if state.get("num_retrive", 0) >= 2:
        print("max tries are done")
        return END
    
    return "route_query"
        

