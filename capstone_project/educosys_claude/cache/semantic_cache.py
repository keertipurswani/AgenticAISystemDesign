from __future__ import annotations

import hashlib
import time

from redis.asyncio import Redis
from redisvl.index import AsyncSearchIndex
from redisvl.query import VectorQuery
from redisvl.query.filter import Tag
from redisvl.redis.utils import array_to_buffer

from educosys_claude.config import config
from educosys_claude.llm.factory import get_embedder
from educosys_claude.observability.logger import get_logger

logger = get_logger(__name__)

VECTOR_DTYPE = "float32"


def _build_index_schema(dims: int) -> dict:
    return {
        "index": {"name": "semantic_cache", "prefix": "cache:"},
        "fields": [
            {
                "name": "query_vector",
                "type": "vector",
                "attrs": {
                    "dims": dims,
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


class SemanticCache:
    """Async Redis-backed semantic cache for /ask responses.

    Embeddings come from the app's own get_embedder() (so the cache follows
    whichever provider config.yaml selects) and lookups filter on both
    domain and model, since this app can switch LLM providers at runtime
    and a cached answer from one model shouldn't be served under another.
    """

    def __init__(self, redis_url: str, threshold: float, dims: int):
        self.client = Redis.from_url(redis_url)
        self.embedder = get_embedder()
        self.threshold = threshold
        self.index = AsyncSearchIndex.from_dict(
            _build_index_schema(dims), redis_client=self.client
        )

    async def setup(self) -> None:
        # overwrite=False: reuse the index if it already exists from a
        # previous run instead of wiping cached entries on every startup.
        await self.index.create(overwrite=False)

    async def _embed(self, text: str) -> list[float]:
        return await self.embedder.aembed_query(text)

    async def _top_match(self, query: str, domain: str, model: str) -> dict | None:
        vector = await self._embed(query)
        q = VectorQuery(
            vector=vector,
            vector_field_name="query_vector",
            return_fields=["response", "query_text", "vector_distance"],
            num_results=1,
            dtype=VECTOR_DTYPE,
        )
        q.set_filter((Tag("domain") == domain) & (Tag("model") == model))
        results = await self.index.query(q)
        return results[0] if results else None

    async def get(self, query: str, domain: str, model: str) -> str | None:
        """Look up a cached response. Returns None on miss."""
        hit = await self._top_match(query, domain=domain, model=model)
        if hit is None:
            return None
        similarity = 1 - float(hit["vector_distance"])
        if similarity >= self.threshold:
            return hit["response"]
        # A candidate exists but isn't close enough - treat as a miss rather
        # than returning a possibly-wrong cached answer.
        return None

    async def put(self, query: str, response: str, domain: str, model: str, ttl: int) -> None:
        """Store a (query, response) pair."""
        vector = await self._embed(query)
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
        await self.index.load([entry], keys=[redis_key], ttl=ttl)

    async def invalidate_domain(self, domain: str) -> None:
        """Delete all cache entries for a domain (e.g. after a codebase change)."""
        async for key in self.client.scan_iter("cache:*"):
            entry = await self.client.hget(key, "domain")
            if entry and entry.decode() == domain:
                await self.client.delete(key)


def get_repo_domain(repo_path: str) -> str:
    """Stable cache domain derived from the repo path, so cached entries
    persist across restarts for the same repo without colliding with
    entries from a different repo indexed by the same tool."""
    return hashlib.sha256(repo_path.encode()).hexdigest()[:16]


async def build_semantic_cache() -> SemanticCache | None:
    """Build and initialize the semantic cache per config.yaml.

    Returns None (and logs why) if caching is disabled or Redis is
    unreachable, so the caller can run without caching instead of crashing.
    """
    cache_config = config.get("semantic_cache", {})
    if not cache_config.get("enabled", False):
        logger.info("Semantic cache: disabled (semantic_cache.enabled is false in config.yaml)")
        return None

    try:
        dims = config["embeddings"]["dims"]
        cache = SemanticCache(
            redis_url=cache_config.get("redis_url", "redis://localhost:6379"),
            threshold=cache_config.get("threshold", 0.85),
            dims=dims,
        )
        await cache.setup()
        logger.info(f"Semantic cache: enabled (threshold={cache.threshold}, dims={dims})")
        return cache
    except Exception as error:
        logger.warning(f"Semantic cache: disabled (init failed: {error})")
        return None