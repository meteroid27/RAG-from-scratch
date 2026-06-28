def reciprocal_rank_fusion(result_lists,k =60):
    
    rrf_scores: dict[str, float] = {}
    doc_lookup: dict[str, dict] = {}  
    
    for results in result_lists:
        for rank, result in enumerate(results):
            doc_key = result["text"]  

            rrf_scores[doc_key] = rrf_scores.get(doc_key, 0) + 1.0 / (rank + k)
            doc_lookup[doc_key] = result
    

    sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    
    return [
        {**doc_lookup[text], "rrf_score": score}
        for text, score in sorted_docs
    ]