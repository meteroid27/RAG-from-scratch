import sys
from pathlib import Path
from tavily import TavilyClient
from huggingface_hub import InferenceClient
from phase_1.reciprocal_fusion import reciprocal_rank_fusion
import os
from dotenv import load_dotenv
load_dotenv() 
sys.path.append(Path(__file__).resolve().parent.parent)
client = InferenceClient()

tavily = TavilyClient(api_key=os.getenv(""))

def evaluate_retrieval_quality(query , chunks):
    context = "\n\n".join(c["text"][:200] for c in chunks[:3])
    
    prompt = f"""Grade how relevant these retrieved passages are to the question.
    Respond with ONLY ONE WORD: "good", "partial", or "bad".

    Question: {query}
    
    Retrieved passages:
    {context}

    Grade (good/partial/bad):"""
    
    response = client.chat.completions.create(
        model = 'Qwen/Qwen2.5-72B-Instruct',
        messages = [{"role":"user", "content":prompt}],
        max_tokens = 10
    )
    
    grade = response.choices[0].message.content.strip().lower()
    
    if grade not in ["good", "partial", "bad"]:
        grade = "partial"
        
    print(f"retrieval grade is: {grade}")
    
    return grade

def web_search_fallback(query):
    results = tavily.search(query=query, max_results=3)
    
    result = [
        {
            "text": r["content"],
            "metadata": {"source": r["url"], "title": r["title"]},
            "score": r.get("score", 0)   
        }
        for r in results['results']
        ]

def crag_retrieval(query, dense_retrieve_fn , bm25_retrieve_fn, chunks):
    dense_result = dense_retrieve_fn(query, k =10)
    bm25_result = bm25_retrieve_fn(query, k =10)
    kb_results = reciprocal_rank_fusion([dense_result,bm25_result])[:5]
    
    grade = evaluate_retrieval_quality(query , kb_results)
    
    if grade == "good":
        print("using the knowledge base results")
        return kb_results    
    
    elif grade == "partial":
        print("Supplementing with web search")
        web_results = web_search_fallback(query)
        return kb_results + web_results
    
    else:
        print("knowledge base failed - using web search only")
        return web_search_fallback(query)    

        
    
    
    

    
