from sentence_transformers import CrossEncoder

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def rerank(query, candidates, top_n):
    if not candidates:
        return []
    
    pairs = [[query, candidate["text"]]for candidate in candidates]
    scores = reranker.predict(pairs)
    
    for candidate, score in zip(candidates, scores):
        candidate["rerank_score"] = float(score)
        
    reranked = sorted(candidates, key=lambda i: i["rerank_score"], reverse=True)
    
    return reranked[:top_n]
    
    
