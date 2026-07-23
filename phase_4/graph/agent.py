# phase4/main.py
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import asyncio
import time
from dotenv import load_dotenv

from phase_1.fetch import fetch_transcript
from phase_1.chunking import chunk_transcript
from phase_1.vector_db import index_chunks, load_indexed_chunks, video_is_indexed
from phase_3.redis_caching import get_cached_answer
from build_graph import build_rag_agent

load_dotenv()

async def ask_agent(video_id: str, query: str) -> str:
    start = time.time()

    cached_answer = get_cached_answer(query, video_id)
    if cached_answer:
        latency = (time.time() - start) * 1000
        print(f"Retries: 0")
        print(f"Grade: cached")
        print(f"Latency: {latency:.0f}ms")
        return cached_answer

    if video_is_indexed(video_id):
        chunks = load_indexed_chunks(video_id)
    else:
        segments = fetch_transcript(video_id)
        chunks = chunk_transcript(segments)
        index_chunks(chunks, video_id)

    agent = build_rag_agent(chunks, video_id)

    final_state = await agent.ainvoke({
        "query": query,
        "documents": [],
        "answer": "",
        "web_search_needed": False,
        "grade": "",
        "num_retries": 0,
        "cache_hit": False
        })

    latency = (time.time() - start) * 1000

    print(f"Retries: {final_state['num_retries']}")
    print(f"Grade: {final_state['grade']}")
    print(f"Latency: {latency:.0f}ms")

    return final_state["answer"]


if __name__ == "__main__":
    answer = asyncio.run(ask_agent(
        video_id="SwQhKFMxmDY",
        query="what are the speakers talking about in this video?"
    ))
    print(answer)