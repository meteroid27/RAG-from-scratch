import asyncio
from concurrent.futures import ThreadPoolExecurter
import sys 
from pathlib import Path
sys.path.append(Path(__file__).resolve().parent.parent)

from phase_1.vector_db import dense_retrival
from phase_1.bm25_search import BM25Index
from phase_1.reciprocal_fusion import reciprocal_rank_fusion

executor = ThreadPoolExecurter(max_workers = 4)

async def async_dense_retrieve(query: str, k: int):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(executor, dense_retrival, query, k)

async def async_bm25_retrieve(query, chunks ,k):
    loop = asyncio.get_event_loop()
    bm25_index = BM25Index(chunks)
    return await loop.run_in_executor(executor, bm25_index.retrieve, query, k)

async def async_hybrid_retrieve(query, chunks , k):
    dense_results , bm25_results = await asyncio.gather(
        async_dense_retrieve(query, k), 
        async_bm25_retrieve(query, chunks, k)
    )
    return reciprocal_rank_fusion([dense_results, bm25_results])

