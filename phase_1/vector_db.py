import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer 

client = chromadb.PersistentClient(path="./chroma_db")

sentence_transformer_ef =  embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name = 'all-MiniLM-L6-v2'
)

collection = client.get_or_create_collection(
    name = "youtube_rag",
    embedding_function= sentence_transformer_ef,
    metadata = {"hnsw:space": "cosine"}

)

def index_chunks(chunks: list[dict], video_id: str):
    ids = [f"{video_id}_chunk_{i}" for i in range(len(chunks))]
    documents = [chunk["text"] for chunk in chunks]
    
    metadatas = [{"start": chunk["start"] , "video_id": video_id} for chunk in chunks]
    
    collection.upsert(documents = documents , ids= ids , metadatas=metadatas)
    print(f"indexed {len(chunks)} chunks for video {video_id}" )
    
def dense_retrival(query , k ):
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

