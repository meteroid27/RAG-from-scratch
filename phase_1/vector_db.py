import chromadb
from chromadb.utils import embedding_functions

client = chromadb.PersistentClient(path="./chroma_db")

_collection = None


def _get_collection():
    global _collection
    if _collection is None:
        sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name='all-MiniLM-L6-v2'
        )
        _collection = client.get_or_create_collection(
            name="youtube_rag",
            embedding_function=sentence_transformer_ef,
            metadata={"hnsw:space": "cosine"}
        )
    return _collection


def video_is_indexed(video_id: str) -> bool:
    collection = _get_collection()
    existing = collection.get(ids=[f"{video_id}_chunk_0"])
    return bool(existing.get("ids"))


def load_indexed_chunks(video_id: str) -> list[dict]:
    collection = _get_collection()
    stored = collection.get(where={"video_id": video_id}, include=["documents", "metadatas"])
    documents = stored.get("documents") or []
    metadatas = stored.get("metadatas") or []

    chunks = []
    for document, metadata in zip(documents, metadatas):
        chunks.append({
            "text": document,
            "start": metadata.get("start", 0),
            "video_id": metadata.get("video_id", video_id),
        })

    return chunks

def index_chunks(chunks: list[dict], video_id: str):
    collection = _get_collection()
    ids = [f"{video_id}_chunk_{i}" for i in range(len(chunks))]
    documents = [chunk["text"] for chunk in chunks]
    
    metadatas = [{"start": chunk["start"] , "video_id": video_id} for chunk in chunks]
    
    collection.upsert(documents = documents , ids= ids , metadatas=metadatas)
    print(f"indexed {len(chunks)} chunks for video {video_id}" )
    
def dense_retrival(query , k ):
    collection = _get_collection()
    results = collection.query(
        query_texts = [query],
        n_results= k,
        include = ["documents","metadatas","distances"]
    )
    
    retrieved = []
    for doc , meta , disc in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0]
    ):
        retrieved.append({
            "text":doc,
            "metadata":meta,
            "score": 1-disc
        })
        
    return retrieved

