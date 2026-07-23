import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import os
from tavily import TavilyClient
from langchain_core.output_parsers import StrOutputParser
from llm.provider import fast_model
from phase_1.reciprocal_fusion import reciprocal_rank_fusion
from dotenv import load_dotenv

load_dotenv()
tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
fast = fast_model()
parser = StrOutputParser()

def evaluate_retrieval_quality(query, chunks):
    context = "\n\n".join(c["text"][:200] for c in chunks[:3])

    prompt = f"""Grade how relevant these retrieved passages are to the question.
    Respond with ONLY ONE WORD: "good", "partial", or "bad".

    Question: {query}

    Retrieved passages:
    {context}

    Grade (good/partial/bad):"""

    response = fast.invoke(prompt)
    grade = parser.invoke(response).strip().lower()

    if grade not in ["good", "partial", "bad"]:
        grade = "partial"

    print(f"retrieval grade is: {grade}")
    return grade


def web_search_fallback(query):
    results = tavily.search(query=query, max_results=3)
    return [
        {
            "text": r["content"],
            "metadata": {"source": r["url"], "title": r["title"]},
            "score": r.get("score", 0)
        }
        for r in results["results"]
    ]