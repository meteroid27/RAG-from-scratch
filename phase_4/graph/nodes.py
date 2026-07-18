import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
import os
import asyncio
from state import AgentState
from  phase_1.bm25_search import BM25Index
from phase_1.vector_db import dense_retrival
from phase_1.reciprocal_fusion import reciprocal_rank_fusion
from phase_1.cross_modal import rerank
from phase_2.crag import web_search_fallback , evaluate_retrieval_quality
from phase_2.multi_query import generate_multi_query
from phase_2.step_back import step_back_query
from phase_2.RRF_multi import RRF_multi_query
from phase_3.redis_caching import get_cached_answer
from phase_3.async_io import  async_bm25_retrieve, async_dense_retrieve, async_hybrid_retrieve
from tavily import TavilyClient
tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
from langchain_huggingface import ChatHuggingFace , HuggingFaceEndpoint
from dotenv import load_dotenv
load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id = "Qwen/Qwen2.5-72B-Instruct",
    task = "text-generation"
)

model =  ChatHuggingFace(llm = llm)


# node 1: router
def route_query(state):
    
    prompt = f"""Decide if this question needs web search or can be answered
    from a local video knowledge base.

    Reply with ONLY one word: "retrieve" or "search"
    
    Question: {state["query"]}"""
    
    response = model.invoke(prompt)
    decision = response.content.strip().lower()
    
    return {
        "web_search_needed": decision == "search",
        "num_retries": state.get("num_retries", 0) 
    }

#node 2: rag retrieval

async def rag_retrieve(state: AgentState, chunks: list[dict]) -> AgentState:
    """
    Step-back + multi-query (your own "rag_fusion" technique), retrieved
    concurrently across all variants, fused with one RRF pass, then CRAG
    grades and corrects before reranking.
    """
    query = state["query"]

    _, step_back = step_back_query(query)
    variants = generate_multi_query(query, n=3)
    all_queries = list(dict.fromkeys([query, step_back] + variants))

    tasks = []
    for q in all_queries:
        tasks.append(async_dense_retrieve(q, k=10))
        tasks.append(async_bm25_retrieve(q, chunks, k=10))
    all_result_lists = await asyncio.gather(*tasks)

    fused = reciprocal_rank_fusion(all_result_lists)[:15]

    grade = evaluate_retrieval_quality(query, fused[:5])
    if grade == "good":
        candidates = fused
    elif grade == "partial":
        candidates = fused + web_search_fallback(query)
    else:
        candidates = web_search_fallback(query)

    final_chunks = rerank(query, candidates[:15])
    return {"documents": final_chunks}

# node 3: web search 
def web_search(state):
    results = tavily.search(query = state["query"], max_results=3)
    
    documents = [
        {
            "text":r["content"],
            "metadata": {"source": r["url"], "title": r["title"]},
            "score": r.get("score", 0)
        }
        for r in results["results"]
        ]
    
    return {"documents": documents}


# node 4: generate
def generate(state):
    context = "\n\n---\n\n".join(
        doc["text"] for doc in state["documents"]
    )
    prompt = f"""Answer the question using ONLY the context provided.
    If the context is insufficient, say so clearly.
    Context:
    {context}
    Question: {state["query"]}
    Answer:"""

    response = model.invoke(prompt)
    return {"answer": response.content}

# node 5: grade + reflect
def grade_and_reflect(state):
    prompt = f"""Grade this answer on two criteria:
    1. Is it faithful to the provided documents? (no hallucination)
    2. Does it actually answer the question?

    Reply with ONLY one word: "good" or "bad"

    Question: {state["query"]}
    Answer: {state["answer"]}"""

    response = model.invoke(prompt)
    grade = response.content.strip().lower()
    if grade not in ["good", "bad"]:
        grade = "bad"

    return {
        "grade": grade,
        "num_retries": state.get("num_retries", 0) + 1
    }


def check_cache(state , video_id):
    cached = get_cached_answer(state["query"], video_id)
    if cached:
        return {"answer" : cached , "cached hit":True}
    return {"cached hit": False}

def cache_result(state , video_id):
    cache_result(state["query"],video_id , state["answer"])
    return {}



    