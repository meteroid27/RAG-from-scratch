import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from phase_1.reciprocal_fusion import reciprocal_rank_fusion

def RRF_multi_query(varient_queries, dense_retrieval , bm25_retrieval , k=10 , rrf_k =60):
    all_result_list = []
    
    for varient in varient_queries:
        dense_result = dense_retrieval(varient, k=k)
        bm25_result = bm25_retrieval(varient, k=k)
        
        if dense_result:
            all_result_list.append(dense_result)
            
        if bm25_result:
            all_result_list.append(bm25_result)
                
    fused = reciprocal_rank_fusion(all_result_list, k = rrf_k)
    
    return fused
    
    
            
        
        