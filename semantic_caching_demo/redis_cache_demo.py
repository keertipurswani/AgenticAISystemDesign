import hashlib
import os
import time

import redis
from openai import OpenAI
from redisvl.index import SearchIndex
from redisvl.query import VectorQuery
from redisvl.query.filter import Tag
from redisvl.redis.utils import array_to_buffer

EMBED_MODEL = "text-embedding-3-small"
EMBED_DIMS = 1536
VECTOR_DTYPE = "float32"

# Defines the Redis index this cache reads/writes: each cached entry is a hash
# with a vector field (for similarity search) plus metadata used to filter results.
INDEX_SCHEMA = {
    "index": {"name": "semantic_cache", "prefix": "cache:"},
    "fields": [
        {
            "name": "query_vector",
            "type": "vector",
            "attrs": {
                "dims": EMBED_DIMS,
                "algorithm": "HNSW",
                "distance_metric": "cosine",
                "datatype": VECTOR_DTYPE,
            },
        },
        {"name": "response", "type": "text"},
        {"name": "query_text", "type": "text"},
        {"name": "domain", "type": "tag"},
        {"name": "model", "type": "tag"},
        {"name": "created_at", "type": "numeric"},
    ],
}

class RedisSemanticCache:
    def __init__(self, redis_url: str | None = None, threshold: float | None = None):
        redis_url = redis_url or os.getenv("REDIS_URL", "redis://localhost:6379")
        threshold = (
            threshold
            if threshold is not None
            else float(os.getenv("SEMANTIC_CACHE_THRESHOLD", "0.92"))
        )

        self.client = redis.from_url(redis_url)
        self.openai = OpenAI()
        self.threshold = threshold
        self.index = SearchIndex.from_dict(INDEX_SCHEMA, redis_client=self.client)
        # overwrite=False: reuse the index if it already exists from a previous run
        # instead of wiping cached entries every time the app starts.
        self.index.create(overwrite=False)

    def _embed(self, text: str) -> list[float]:
        # Converts text to a vector so semantically similar questions land close
        # together in vector space, even when worded completely differently.
        return self.openai.embeddings.create(
            model=EMBED_MODEL, input=text
        ).data[0].embedding

    def _top_match(self, query: str, domain: str = "*") -> dict | None:
        # Shared by similarity() and get() so both use the exact same nearest-
        # neighbor lookup instead of duplicating the query logic.
        vector = self._embed(query)
        q = VectorQuery(
            vector=vector,
            vector_field_name="query_vector",
            return_fields=["response", "query_text", "vector_distance"],
            num_results=1,
            dtype=VECTOR_DTYPE,
        )
        if domain != "*":
            # Scope the search to one domain so unrelated demos/topics sharing
            # the same Redis instance can't match each other's cached answers.
            q.set_filter(Tag("domain") == domain)

        results = self.index.query(q)
        return results[0] if results else None

    def similarity(self, query: str, domain: str = "*", model: str = "*") -> float | None:
        """Return the top candidate's similarity score, or None if no candidate exists."""
        hit = self._top_match(query, domain=domain)
        if hit is None:
            return None
        # Redis returns cosine distance (0 = identical); convert to similarity
        # (1 = identical) so it's directly comparable to self.threshold.
        return 1 - float(hit["vector_distance"])

    def get(self, query: str, domain: str = "*", model: str = "*") -> str | None:
        """Look up a cached response. Returns None on miss."""
        hit = self._top_match(query, domain=domain)
        if hit is None:
            return None

        similarity = 1 - float(hit["vector_distance"])
        if similarity >= self.threshold:
            return hit["response"]
        # A candidate exists but isn't close enough - treat it as a miss rather
        # than returning a possibly-wrong cached answer.
        return None

    def put(
        self,
        query: str,
        response: str,
        domain: str = "general",
        model: str = "gpt-4o",
        ttl: int = 86400,
    ) -> str:
        """Store a (query, response) pair. Returns the cache key."""
        vector = self._embed(query)
        # Every Redis entry needs a key - that's a Redis requirement, unrelated to
        # the semantic (vector similarity) matching itself, which happens in
        # _top_match() regardless of how this key is named. Hashing the exact query
        # text just keeps the key stable/short and means storing the exact same
        # question twice overwrites rather than piling up duplicate entries.
        doc_id = hashlib.sha256(query.encode()).hexdigest()[:16]
        redis_key = self.index.key(doc_id)
        entry = {
            "query_vector": array_to_buffer(vector, VECTOR_DTYPE),
            "response": response,
            "query_text": query,
            "domain": domain,
            "model": model,
            "created_at": time.time(),
        }
        # ttl expires the entry automatically so stale cached answers don't
        # accumulate forever without manual cleanup.
        loaded_keys = self.index.load([entry], keys=[redis_key], ttl=ttl)
        return loaded_keys[0]

    def invalidate_domain(self, domain: str):
        """Delete all cache entries for a domain (e.g. after knowledge update)."""
        # redisvl has no server-side "delete by tag filter" here, so entries are
        # scanned and checked one by one against the requested domain.
        for key in self.client.scan_iter("cache:*"):
            entry = self.client.hget(key, "domain")
            if entry and entry.decode() == domain:
                self.client.delete(key)
