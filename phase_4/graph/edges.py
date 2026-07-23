from langgraph.graph import END
from state import AgentState

def decide_retrieve_path(state):
    path = state["retrieval_path"]
    
    if path == "direct":
        return "generate"
    if  path == "search":
        return   "web_search"
    if path == "simple":
        return "simple_rag_retrieve"
    else: 
        return "rag_retrieve"
    
def decide_after_grading(state):
    
    if state["grade"] == "good":
        return "cache_result"
    
    if state.get("num_retrive", 0) >= 2:
        print("max tries are done")
        return END
    
    return "route_query"
        

