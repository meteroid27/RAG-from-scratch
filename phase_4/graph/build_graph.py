from functools import partial
from langgraph.graph import StateGraph , END
from graph.state import AgentState
from graph.nodes import route_query, rag_retrieve, web_search, generate, grade_and_reflect
from graph.edges import decide_retrieve_path , decide_after_grading


def rag_retrieve_agent(chunks):
    
    graph = StateGraph(AgentState)
    
    graph.add_node("route_query", route_query)
    graph.add_node("rag_retrieve",     partial(rag_retrieve, chunks=chunks))
    graph.add_node("web_search", web_search)
    graph.add_node("generate", generate)
    graph.add_node("grade_and_reflect", grade_and_reflect)
    
    
    graph.set_entry_point("route_query")
    
    graph.add_conditional_edges(
        "route_edges",
        decide_retrieve_path,
        {
            "web_search":"web_search",
            "rag_retrieve":"rag_retrieve"
        }
    )
    
    graph.add_edge("web_search", "generate")
    graph.add_edge("rag_retrieve", "generate")
    graph.add_edge("generate","grade_and_reflect")
    
    graph.add_conditional_edges(
        "grade_and_reflect",
        decide_after_grading,
        {
            END: END, 
            "route_query":"route_query"
        }
        
    )
    return graph.compile()