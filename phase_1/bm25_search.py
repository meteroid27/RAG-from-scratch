from rank_bm25 import BM25Okapi
import re

def tokenize(text: str) -> list[str]:
    """Simple tokenizer. In production you'd use nltk or spacy."""
    return re.findall(r'\b\w+\b', text.lower())

class BM25Index:
    def __init__(self, chunks: list[dict]):
        self.chunks = chunks
        tokenized_corpus = [tokenize(chunk["text"]) for chunk in chunks]
        self.bm25 = BM25Okapi(tokenized_corpus)
    
    def retrieve(self, query: str, k: int = 10) -> list[dict]:
        """Return top-k chunks ranked by BM25 score."""
        tokenized_query = tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)
        
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [
            {
                "text": self.chunks[idx]["text"],
                "metadata": self.chunks[idx],
                "score": float(scores[idx])
            }
            for idx in top_indices
            if scores[idx] > 0  
        ]