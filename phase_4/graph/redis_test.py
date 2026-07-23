import os
from dotenv import load_dotenv
import redis

load_dotenv()

r = redis.Redis(
    host=os.environ["REDIS_HOST"].strip(),
    port=int(os.environ["REDIS_PORT"].strip()),
    password=os.environ["REDIS_PASSWORD"].strip(),
    ssl=False,
    ssl_cert_reqs=None,
    decode_responses=True,
)

print(r.ping())