import os
from typing import Any

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.sqlite import SqliteSaver

DEFAULT_MODEL_NAME = "gpt-4o"

def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name, str(default)).strip().lower()
    return value in {"1", "true", "yes", "on"}

def _build_semantic_cache():
    from redis_cache_demo import RedisSemanticCache

    return RedisSemanticCache(
        redis_url=os.getenv("REDIS_URL", "redis://localhost:6379"),
        threshold=float(os.getenv("SEMANTIC_CACHE_THRESHOLD", "0.80")),
    )

def _last_message_text(result: dict[str, Any]) -> str:
    return str(result["messages"][-1].content)

def _build_agent(model, checkpointer):
    return create_agent(
        model=model,
        tools=[],
        system_prompt=(
            "You are a concise helpful assistant. "
            "Answer clearly in a few sentences unless the user asks for detail."
        ),
        checkpointer=checkpointer,
    )

def main() -> None:
    load_dotenv(override=True)

    model_name = os.getenv("CACHE_DEMO_MODEL_NAME", DEFAULT_MODEL_NAME)
    model = init_chat_model(
        model=model_name,
        model_provider="openai",
        temperature=0,
    )

    db_path = os.getenv("CACHE_DEMO_SQLITE_CHECKPOINT_PATH", "data/cache_demo_memory.db")
    db_dir = os.path.dirname(db_path)
    if db_dir:
        os.makedirs(db_dir, exist_ok=True)
    thread_id = os.getenv("CACHE_DEMO_CHAT_THREAD_ID", "cache-demo-thread-1")
    
    cache_domain = os.getenv("SEMANTIC_CACHE_DOMAIN", "cache-demo")
    cache_model_name = model_name

    semantic_cache = None
    if _env_bool("SEMANTIC_CACHE_ENABLED", default=False):
        try:
            semantic_cache = _build_semantic_cache()
            print("Semantic cache: enabled (Redis vector similarity)")
            print(f"Cache domain: {cache_domain} | model tag: {cache_model_name}")
            print(f"Cache threshold: {semantic_cache.threshold}")
        except Exception as error:
            # Redis being unreachable (or unconfigured) shouldn't stop the demo -
            # it just runs without caching.
            print(f"Semantic cache: disabled (init failed: {error})")
    else:
        print("Semantic cache: disabled (set SEMANTIC_CACHE_ENABLED=true in .env)")

    print("Semantic Cache Demo (LangChain)")
    print("Type 'exit' to quit. Type 'clear-cache' to invalidate cached answers.\n")
    print(f"Thread ID: {thread_id}")
    print(f"Checkpoint DB: {db_path}")
    print("Try asking the same question twice in different words to see a cache HIT.\n")
    print("Example:")
    print("- What is semantic caching?")
    print("- Explain semantic cache in simple terms.\n")

    with SqliteSaver.from_conn_string(db_path) as checkpointer:
        agent = _build_agent(model, checkpointer=checkpointer)

        while True:
            user_input = input("You: ").strip()
            if not user_input:
                continue
            if user_input.lower() in {"exit", "quit"}:
                print("Bye!")
                break
            if user_input.lower() == "clear-cache":
                if semantic_cache is None:
                    print("Semantic cache is not enabled.\n")
                    continue
                try:
                    semantic_cache.invalidate_domain(cache_domain)
                    print(f"[Cache] Cleared entries for domain '{cache_domain}'.\n")
                except Exception as error:
                    print(f"[Cache] clear failed: {error}\n")
                continue

            if semantic_cache is not None:
                try:
                    top_similarity = semantic_cache.similarity(
                        user_input,
                        domain=cache_domain,
                        model=cache_model_name,
                    )
                    if top_similarity is None:
                        print(
                            f"[Cache] similarity: no prior candidate | threshold={semantic_cache.threshold:.2f}"
                        )
                    else:
                        predicted = (
                            "HIT"
                            if top_similarity >= semantic_cache.threshold
                            else "MISS"
                        )
                        print(
                            f"[Cache] similarity={top_similarity:.4f} | threshold={semantic_cache.threshold:.2f} | predicted={predicted}"
                        )

                    cached_response = semantic_cache.get(
                        user_input,
                        domain=cache_domain,
                        model=cache_model_name,
                    )
                except Exception as error:
                    # Treat any cache error as a MISS rather than crashing the
                    # chat loop - the LLM call below is always the fallback.
                    cached_response = None
                    print(f"[Cache] lookup failed: {error}")
                else:
                    if cached_response:
                        # On a HIT we skip the LLM call entirely and reuse the
                        # stored answer, which is the whole point of the cache.
                        print("\nAssistant (semantic cache HIT):")
                        print(cached_response)
                        print()
                        continue
                    print("[Cache] MISS - calling the LLM")

            # Reached on a cache MISS, a cache error, or when caching is disabled.
            # thread_id ties this call into the SqliteSaver checkpoint so the
            # agent remembers earlier turns in the same conversation.
            result = agent.invoke(
                {"messages": [{"role": "user", "content": user_input}]},
                config={"configurable": {"thread_id": thread_id}},
            )
            response_text = _last_message_text(result)
            print(f"\nAssistant:\n{response_text}\n")

            if semantic_cache is not None:
                try:
                    # Store the fresh answer so a future, differently-worded
                    # version of this question can be served from cache.
                    semantic_cache.put(
                        user_input,
                        response_text,
                        domain=cache_domain,
                        model=cache_model_name,
                        ttl=int(os.getenv("SEMANTIC_CACHE_TTL", "86400")),
                    )
                    print("[Cache] Stored response for future semantic matches.\n")
                except Exception as error:
                    print(f"[Cache] store failed: {error}\n")

if __name__ == "__main__":
    main()
