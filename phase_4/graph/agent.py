import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from phase_1.chunking import chunk_transcript
from phase_1.vector_db import index_chunks
from phase_1.fetch import fetch_transcript
from build_graph import build_rag_agent

def ask_agent(video_id , query):
    segment = fetch_transcript(video_id)
    chunks = chunk_transcript(segment)
    index_chunks(chunks, video_id)
    
    agent = build_rag_agent(chunks)
    
    final_state = agent.invoke({
        "query": query,
        "documents": [],
        "answer": "",
        "web_search_needed": False,
        "grade": "",
        "num_retries": 0
    })
    
    
    print(f"Retries needed: {final_state['num_retries']}")
    print(f"Final grade: {final_state['grade']}")
    return final_state["answer"]


if __name__ == "__main__":
        answer = ask_agent(
        video_id="SwQhKFMxmDY",
        query="what did they say about the dopamine system?"
        )
        print(answer)
    
    
    