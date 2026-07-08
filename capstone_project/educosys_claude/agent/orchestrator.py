from educosys_claude.config import config
from educosys_claude.observability.logger import get_logger


logger = get_logger(__name__)


async def handle_query(
    agent,
    question: str,
    thread_id: str,
    semantic_cache=None,
    cache_domain: str | None = None,
) -> str:
    """Entry point for all user queries - invokes the agent.

    When semantic_cache/cache_domain are provided, a cache hit returns the
    stored answer directly and skips the agent (and every tool call it would
    have made, including search_codebase) entirely. A miss falls through to
    the normal agent call and stores the fresh answer for next time.
    """
    logger.info(f"Handling query for session {thread_id}: {question}")
    model = config["llm"]["model"]

    if semantic_cache is not None and cache_domain is not None:
        try:
            cached_response = await semantic_cache.get(question, domain=cache_domain, model=model)
        except Exception as e:
            logger.warning(f"Semantic cache lookup failed, falling back to agent: {e}")
            cached_response = None
        if cached_response is not None:
            logger.info("Semantic cache HIT - skipping agent/tool calls")
            return cached_response

    agent_config = {"configurable": {"thread_id": thread_id}}
    try:
        response = await agent.ainvoke(
            {"messages": [{"role": "user", "content": question}]}, agent_config
        )
        answer = response["messages"][-1].content
    except Exception as e:
        logger.error(f"Agent error: {e}")
        return f"Error: {e}"

    if semantic_cache is not None and cache_domain is not None:
        try:
            ttl = config.get("semantic_cache", {}).get("ttl", 86400)
            await semantic_cache.put(question, answer, domain=cache_domain, model=model, ttl=ttl)
        except Exception as e:
            logger.warning(f"Semantic cache store failed: {e}")

    return answer