import redis
import hashlib
import json

redis_client = redis.Redis(host="localhost", port=6379, decode_responses=True)
CACHE_TTL = 3600 

def make_cache_key(query: str, video_id: str):
    content = f"{video_id}:{query.lower().strip()}"
    return f"rag:{hashlib.md5(content.encode()).hexdigest()}"

def get_cached_answer(query: str, video_id: str) :
    """Return cached answer if it exists, None otherwise."""
    key = make_cache_key(query, video_id)
    cached = redis_client.get(key)
    if cached:
        print(f"Cache HIT for query: {query[:50]}...")
        return json.loads(cached)
    print(f"Cache MISS for query: {query[:50]}...")
    return None

def cache_answer(query: str, video_id: str, answer: str):
    """Store the answer in Redis with a TTL."""
    key = make_cache_key(query, video_id)
    redis_client.setex(key, CACHE_TTL, json.dumps(answer))
    print(f"Cached answer for: {query[:50]}...")