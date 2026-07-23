# rag_phase1.py
import logging
import os
from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace , HuggingFaceEndpoint
from fetch import fetch_transcript
from chunking import chunk_transcript
from vector_db import collection, index_chunks, dense_retrival
from bm25_search import BM25Index
from reciprocal_fusion import reciprocal_rank_fusion
from cross_modal import rerank
'''
from regax import evaluate_rag_system
'''

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Config ───────────────────────────────────────────────
VIDEO_ID = "SwQhKFMxmDY"
RRF_K = 60
DENSE_K = 15
BM25_K = 15

# ── Generate ─────────────────────────────────────────────
def generate(query: str, context_chunks: list[dict]) -> str:
    context = "\n\n---\n\n".join(c["text"] for c in context_chunks)
    prompt = f"""Answer the question using ONLY the context provided.
If the context does not contain enough information, say so clearly.

Context:
{context}

Question: {query}

Answer:"""
    llm = HuggingFaceEndpoint(
        repo_id = 'Qwen/Qwen2.5-72B-Instruct',
        task = 'text-generation'
    )
    
    model = ChatHuggingFace(llm=llm)
    return model.invoke(prompt).content

    '''
    llm = ChatOpenAI(model="gpt-4o-mini")
    return llm.invoke(prompt).content\
    '''

# ── Main pipeline ─────────────────────────────────────────
def ask(video_id: str, query: str) -> str:
    logger.info(f"Query: {query}")

    # Step 1: Fetch and chunk
    segments = fetch_transcript(video_id)
    for x in range(5):
        print(f"the segments is: {segments[x]}")
    chunks = chunk_transcript(segments)

    # Step 2: Index into ChromaDB
    index_chunks(chunks, video_id)

    # Step 3: Hybrid retrieval
    dense_results = dense_retrival(query, k=DENSE_K)
    
    bm25_index = BM25Index(chunks)
    bm25_results = bm25_index.retrieve(query, k=BM25_K)
    logger.info(f"Dense: {len(dense_results)} | BM25: {len(bm25_results)}")

    # Step 4: Fuse with RRF
    fused = reciprocal_rank_fusion([dense_results, bm25_results])
    logger.info(f"After RRF: {len(fused)} unique candidates")

    # Step 5: Rerank top 15
    final_chunks = rerank(query, fused, 15)
    logger.info(f"Final chunks sent to LLM: {len(final_chunks)}")

    # Step 6: Generate answer
    return generate(query, final_chunks)


if __name__ == "__main__":
    query = "what did they say about the dopamine system?"
    answer = ask(VIDEO_ID, query)
    print(answer)
    