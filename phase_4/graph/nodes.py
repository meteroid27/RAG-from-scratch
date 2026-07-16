import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
import os
from  phase_1.bm25_search import BM25Index
from phase_1.vector_db import dense_retrival
from phase_1.reciprocal_fusion import reciprocal_rank_fusion
from phase_1.cross_modal import rerank
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
        "web search needed": decision == "search",
        "num_retries": state.get("num_retries", 0) 
    }

#node 2: rag retrieval
def rag_retrieve(state, chunks):
    query = state["query"]
    dense_result = dense_retrival(query , k = 15)
    bm25_index = BM25Index(chunks )
    bm25_result = bm25_index.retrieve(query , k=15)
    fused = reciprocal_rank_fusion([dense_result, bm25_result ])
    final_chunks = rerank(query, fused[:15])
    
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
