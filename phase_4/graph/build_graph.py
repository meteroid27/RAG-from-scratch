from functools import partial
from langgraph.graph import StateGraph , END
from state import AgentState
from nodes import route_query, rag_retrieve, web_search, generate, grade_and_reflect, check_cache , cache_result , simple_rag_retrieve
from edges import decide_retrieve_path , decide_after_grading


def build_rag_agent(chunks, video_id):
    
    graph = StateGraph(AgentState)
    
    graph.add_node("check_cache", partial(check_cache , video_id = video_id))
    graph.add_node("route_query", route_query)
    graph.add_node("rag_retrieve", partial(rag_retrieve, chunks=chunks))
    graph.add_node("web_search", web_search)
    graph.add_node("generate", generate)
    graph.add_node("grade_and_reflect", grade_and_reflect)
    graph.add_node("cache_result", partial(cache_result, video_id=video_id))
    graph.add_node("simple_rag_retrieve", partial(simple_rag_retrieve , chunks = chunks))
    
    
    graph.set_entry_point("check_cache")

    graph.add_conditional_edges(
        "check_cache",
        lambda state: END if state["cache_hit"] else "route_query",
        {END: END, "route_query": "route_query"}
        )
    
    graph.add_conditional_edges(
        "route_query", decide_retrieve_path,
        {
            "rag_retrieve":"rag_retrieve",
            "web_search":"web_search",
            "simple_rag_retrieve": "simple_rag_retrieve",
            "generate":"generate"
        }
    )
    graph.add_edge("simple_rag_retrieve","generate")
    graph.add_edge("rag_retrieve", "generate")
    graph.add_edge("web_search","generate")
    graph.add_edge("generate","grade_and_reflect")
    graph.add_conditional_edges(
        "grade_and_reflect", decide_after_grading,
        {"cache_result": "cache_result", "route_query": "route_query", END: END}
        )
    graph.add_edge("cache_result", END)

    return graph.compile()