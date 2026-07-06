"""Live demo: implicit prompt caching + token budgeting against the real OpenAI API (GPT-4o).

Unlike Anthropic, OpenAI's caching has no cache_control param and no separate
"write" price tier — caching is automatic for prompts >= 1024 tokens, matched
by prefix hash, and cached tokens simply bill cheaper on any request that
happens to hit. There's no way to force a hit or to know in advance whether
one will land; `prompt_cache_key` only nudges routing to improve hit odds.
"""

import os
import sys
from dataclasses import dataclass, field

from dotenv import load_dotenv
from openai import OpenAI

MODEL = "gpt-4o"
MIN_CACHEABLE_TOKENS = 1024  # OpenAI's cache floor; shorter prompts never cache
CACHE_KEY = "educosys-caching-demo-x"  # routing hint, not a guarantee

# Per-million-token prices (USD) used to compute both the real spend and the
# hypothetical "no cache" comparison printed at the end of the demo.
# gpt-4o's cache discount is 50% off input (vs. gpt-5.4's ~90% off).
PRICE_PER_MTOK = {
    "input": 2.50,
    "cached_input": 1.25,
    "output": 10.00,
}

# Each question is sent as a separate request that reuses the same static
# prefix (system prompt + tool defs + KB), so later requests are the ones
# eligible for an implicit cache hit on that shared prefix.
QUESTIONS = [
    "What's the rate limit per API key?",
    "What should a client do when it gets a 429?",
    "What's the request timeout for the gateway?",
    "How are deployments rolled out across regions?",
    "Which tool searches the codebase?",
    "Which tool triggers a CI pipeline run?",
    "How does the gateway authenticate requests?",
    "Which tool opens a pull request?",
    "Summarize the gateway's reliability guarantees in one sentence.",
    "How many regions does a canary deployment cover?",
]


class BudgetExceeded(Exception):
    pass


@dataclass
class TokenBudget:
    max_usd: float
    spent_usd: float = 0.0
    log: list = field(default_factory=list)  # (label, usage, cost) per charged request, for the summary stats

    def charge(self, label, usage):
        # OpenAI reports cached tokens as a subset of prompt_tokens, not a separate
        # count, so the non-cached portion has to be derived by subtraction.
        cached = usage.prompt_tokens_details.cached_tokens or 0
        uncached_input = usage.prompt_tokens - cached
        cost = (
            uncached_input * PRICE_PER_MTOK["input"]
            + cached * PRICE_PER_MTOK["cached_input"]
            + usage.completion_tokens * PRICE_PER_MTOK["output"]
        ) / 1_000_000
        # Budget check happens after the cost is known but before it's recorded,
        # so an over-budget request raises instead of silently being counted.
        if self.spent_usd + cost > self.max_usd:
            raise BudgetExceeded(
                f"{label} would push spend to ${self.spent_usd + cost:.4f}, "
                f"over the ${self.max_usd:.2f} budget — stopping before the call is sent"
            )
        self.spent_usd += cost
        self.log.append((label, usage, cost))
        return cost


def build_static_prefix(repeat):
    # Same KB paragraph repeated `repeat` times so the caller can pad the prefix
    # past MIN_CACHEABLE_TOKENS (real KBs wouldn't need this padding trick).
    system_prompt = (
        "You are an internal assistant for the Platform team. Answer "
        "only using the API reference and tool definitions below. Be concise and "
        "cite the relevant section number when possible.\n\n"
    )
    tool_defs = (
        "Available tools (described for reference; not invoked in this demo):\n"
        "- search_codebase(query): full-text search across the monorepo\n"
        "- run_ci_pipeline(pipeline_id): trigger a CI pipeline run\n"
        "- open_pull_request(branch, title, description): open a PR\n\n"
    )
    kb_chunk = (
        "Section {n}: The API Gateway enforces a rate limit of 1000 requests/minute "
        "per API key; clients should back off exponentially on 429 responses. "
        "Requests time out after 30 seconds. All endpoints require a bearer token "
        "issued by the Auth service. Deployments are canary-rolled over 15 minutes "
        "across 3 regions before going fully live. "
    )
   # knowledge_base = "".join(kb_chunk.format(n=i) for i in range(repeat))
    static_prefix = system_prompt + tool_defs + kb_chunk

    try:
        # tiktoken gives an exact count matching what the API will bill for.
        import tiktoken

        enc = tiktoken.encoding_for_model(MODEL)
        token_count = len(enc.encode(static_prefix))
    except Exception:
        token_count = len(static_prefix) // 4  # rough fallback if tiktoken/model encoding is unavailable

    return static_prefix, token_count


def run_demo(repeat, budget_usd, max_output_tokens):
    client = OpenAI()
    static_prefix, prefix_tokens = build_static_prefix(repeat)

    print(f"Static prefix (system prompt + tool defs + KB, sent on every request): ~{prefix_tokens} tokens")
    if prefix_tokens < MIN_CACHEABLE_TOKENS:
        print(
            f"  warning: below the {MIN_CACHEABLE_TOKENS}-token cache floor for "
            f"{MODEL} — raise KB_REPEAT or caching will never trigger"
        )
    print(f"Budget: ${budget_usd:.2f}  |  model: {MODEL}\n")

    budget = TokenBudget(max_usd=budget_usd)

    # prompt = total input tokens for that request (static prefix + question);
    # cached = how many of those were served from OpenAI's cache at the cheaper
    # rate; %cach = cached / prompt; output = tokens in the reply.
    print(
        "prompt = total input tokens sent | cached = of those, how many hit the "
        "cache (billed cheaper) | output = tokens in the reply\n"
    )

    # Column widths, reused for both the header and every data row so they
    # line up; joined with " | " since same-width headers alone (e.g. "cached"
    # next to "prompt") are hard to tell apart without a separator.
    COLS = [("#", 2), ("cache", 5), ("cached", 6), ("prompt", 6), ("%cache", 5), ("output", 6), ("cost", 9)]

    def row(*values):
        return " | ".join(f"{v:>{w}}" for v, (_, w) in zip(values, COLS))

    header = row(*(title for title, _ in COLS))
    print(header)
    print("-" * len(header))

    for i, question in enumerate(QUESTIONS, start=1):
        # The static_prefix (system message) is identical on every request, so
        # OpenAI's implicit prefix cache can potentially reuse it after the
        # first request warms it up — but a hit is never guaranteed here.
        response = client.chat.completions.create(
            model=MODEL,
            max_completion_tokens=max_output_tokens,
            prompt_cache_key=CACHE_KEY,
            messages=[
                {"role": "system", "content": static_prefix},
                {"role": "user", "content": question},
            ],
        )
        usage = response.usage
        cost = budget.charge(f"request {i}", usage)  # raises BudgetExceeded before printing if over budget
        cached = usage.prompt_tokens_details.cached_tokens or 0
        hit = "hit" if cached else "miss"
        pct_cached = (cached / usage.prompt_tokens * 100) if usage.prompt_tokens else 0.0

        print(row(i, hit, cached, usage.prompt_tokens, f"{pct_cached:.0f}%", usage.completion_tokens, f"${cost:.4f}"))

    # Recompute what the same requests would have cost if every prompt token
    # were billed at the full (non-cached) input rate, as a baseline to compare against.
    no_cache_total = sum(
        usage.prompt_tokens * PRICE_PER_MTOK["input"] / 1_000_000
        + usage.completion_tokens * PRICE_PER_MTOK["output"] / 1_000_000
        for _, usage, _ in budget.log
    )

    savings = no_cache_total - budget.spent_usd
    pct = (savings / no_cache_total * 100) if no_cache_total else 0.0

    # Break actual spend down by rate tier, so it's visible *why* the overall
    # savings % is lower than the %cached seen per-request: output tokens are
    # never cached, and gpt-4o's cache discount is 50% off (not free), not 100%.
    total_cached_tok = sum(usage.prompt_tokens_details.cached_tokens or 0 for _, usage, _ in budget.log)
    total_prompt_tok = sum(usage.prompt_tokens for _, usage, _ in budget.log)
    total_uncached_tok = total_prompt_tok - total_cached_tok
    total_output_tok = sum(usage.completion_tokens for _, usage, _ in budget.log)
    num_hits = sum(1 for _, usage, _ in budget.log if usage.prompt_tokens_details.cached_tokens)

    cached_cost = total_cached_tok * PRICE_PER_MTOK["cached_input"] / 1_000_000
    uncached_cost = total_uncached_tok * PRICE_PER_MTOK["input"] / 1_000_000
    output_cost = total_output_tok * PRICE_PER_MTOK["output"] / 1_000_000

    print()
    print(f"Cache hit rate: {num_hits}/{len(budget.log)} requests")
    print("Where the actual spend went:")
    print(
        f"  input, cached:    {total_cached_tok:>6,} tok @ ${PRICE_PER_MTOK['cached_input']:.2f}/1M "
        f"= ${cached_cost:.4f}   (this is what caching made cheaper)"
    )
    print(
        f"  input, uncached:  {total_uncached_tok:>6,} tok @ ${PRICE_PER_MTOK['input']:.2f}/1M "
        f"= ${uncached_cost:.4f}   (cold-start miss + non-cached remainder)"
    )
    print(
        f"  output:           {total_output_tok:>6,} tok @ ${PRICE_PER_MTOK['output']:.2f}/1M "
        f"= ${output_cost:.4f}   (never eligible for caching, at any hit rate)"
    )
    print()
    print(f"Actual spend (with caching):     ${budget.spent_usd:.4f}")
    print(f"Spend if every request had missed: ${no_cache_total:.4f}  (same tokens, all billed at the full input rate)")
    print(f"Savings from caching:             ${savings:.4f}  ({pct:.1f}% cheaper)")
    print(
        f"\nNote: caching only discounts input tokens (${PRICE_PER_MTOK['input']:.2f} -> "
        f"${PRICE_PER_MTOK['cached_input']:.2f} per 1M), not output — so overall savings "
        "will always trail the %cached seen per-request. Unlike Claude, every 'hit' "
        "above was also opportunistic: OpenAI gives no way to guarantee or "
        "pre-verify a cache hit before sending."
    )

if __name__ == "__main__":
    load_dotenv(override=True)  # loads OPENAI_API_KEY (and any overrides below) from .env

    # All three knobs are env-var configurable so the demo can be tuned (e.g.
    # to push the prefix past MIN_CACHEABLE_TOKENS) without editing code.
    repeat = int(os.getenv("KB_REPEAT", "30"))
    budget_usd = float(os.getenv("DEMO_BUDGET_USD", "2.00"))
    max_output_tokens = int(os.getenv("DEMO_MAX_OUTPUT_TOKENS", "200"))
    
    try:
        run_demo(repeat, budget_usd, max_output_tokens)
    except BudgetExceeded as e:
        print(f"\nStopped: {e}")
        sys.exit(1)
