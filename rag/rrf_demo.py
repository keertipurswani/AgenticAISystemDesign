# ---------------------------------------------------------------------------
# rrf_demo.py — How Reciprocal Rank Fusion works, step by step
#
# No LLM, no embeddings. Pure Python so every number is visible.
# ---------------------------------------------------------------------------

# Imagine a user asks: "How does the app enforce task limits per user?"
#
# Two retrievers independently search the codebase and return ranked lists.
# Neither list is perfect on its own — RRF fuses them into one better list.

QUERY = "How does the app enforce task limits per user?"

# Semantic retriever results — understands concepts, misses exact identifiers
SEMANTIC_RESULTS = [
    "billing.py::BillingService",        # rank 1 — talks about limits conceptually
    "task_service.py::create_task",      # rank 2 — mentions task creation flow
    "permissions.py::PermissionPolicy",  # rank 3 — access control concept
    "auth.py::AuthService",              # rank 4 — loosely related
]

# Lexical (BM25) results — exact token matches, misses conceptual docs
LEXICAL_RESULTS = [
    "billing.py::check_task_limit",      # rank 1 — exact words 'task' + 'limit'
    "billing.py::TASK_LIMITS",           # rank 2 — exact token 'TASK_LIMITS'
    "models.py::Task",                   # rank 3 — exact token 'task'
    "task_service.py::create_task",      # rank 4 — exact token 'task'
]

RRF_K = 60   # smoothing constant — prevents rank-1 from dominating too heavily


# ---------------------------------------------------------------------------
# STEP 1 — Show both ranked lists side by side
# ---------------------------------------------------------------------------

print(f'  Query: "{QUERY}"')

print()
print("--- STEP 1: What each retriever returned on its own ---")
print()
print(f"  {'Rank':<6}  {'Semantic results':<35}  {'Lexical (BM25) results'}")
print(f"  {'-'*6}  {'-'*35}  {'-'*35}")
for i, (s, l) in enumerate(zip(SEMANTIC_RESULTS, LEXICAL_RESULTS), 1):
    print(f"  {i:<6}  {s:<35}  {l}")

print()
print("  Problem:")
print("  • Semantic missed 'check_task_limit' (exact function name)")
print("  • Lexical missed 'BillingService'    (conceptual match, no exact tokens)")
print("  • 'create_task' appears in both — a signal it's probably relevant")
print()
print("  RRF fuses both lists so neither retriever's blind spots dominate.")


# ---------------------------------------------------------------------------
# STEP 2 — Compute the RRF score for each result
# ---------------------------------------------------------------------------

print()
print("--- STEP 2: RRF formula ---")
print()
print("  RRF score  =  Σ  1 / (k + rank)")
print("               (summed over every retriever that returned this chunk)")
print()
print(f"  k = {RRF_K}  ← smoothing constant")
print()
print("  Why 1/(k+rank)?")
print("  • Rank 1  →  1/(60+1) = 0.0164   (high reward)")
print("  • Rank 2  →  1/(60+2) = 0.0161   (slightly less)")
print("  • Rank 4  →  1/(60+4) = 0.0156   (still rewarded, just less)")
print()
print("  A chunk found by BOTH retrievers gets TWO terms added together,")
print("  which is why agreement across retrievers pushes a chunk to the top.")


# ---------------------------------------------------------------------------
# STEP 3 — Calculate scores for every chunk
# ---------------------------------------------------------------------------

print()
print("--- STEP 3: Calculating scores ---")
print()

scores = {}
source_map = {}

def add_scores(results, label):
    for rank, chunk in enumerate(results, 1):
        rrf = 1 / (RRF_K + rank)
        scores[chunk] = scores.get(chunk, 0.0) + rrf
        source_map.setdefault(chunk, []).append((label, rank, round(rrf, 6)))

add_scores(SEMANTIC_RESULTS, "semantic")
add_scores(LEXICAL_RESULTS,  "lexical")

print(f"  {'Chunk':<40}  {'Retriever':<10}  {'Rank':<6}  {'1/(k+rank)':<12}  {'Running total'}")
print(f"  {'-'*40}  {'-'*10}  {'-'*6}  {'-'*12}  {'-'*13}")

running = {}
for chunk in list(dict.fromkeys(SEMANTIC_RESULTS + LEXICAL_RESULTS)):  # preserve order
    for label, rank, rrf in source_map.get(chunk, []):
        running[chunk] = running.get(chunk, 0.0) + rrf
        print(f"  {chunk:<40}  {label:<10}  {rank:<6}  {rrf:<12.6f}  {running[chunk]:.6f}")


# ---------------------------------------------------------------------------
# STEP 4 — Final fused ranking
# ---------------------------------------------------------------------------

print()
print("--- STEP 4: Final fused ranking ---")
print()

ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

print(f"  {'Rank':<6}  {'RRF score':<12}  {'Found by':<25}  Chunk")
print(f"  {'-'*6}  {'-'*12}  {'-'*25}  {'-'*40}")

for final_rank, (chunk, score) in enumerate(ranked, 1):
    retrievers = " + ".join(label for label, _, _ in source_map[chunk])
    star = "  ◄ both agreed!" if len(source_map[chunk]) == 2 else ""
    print(f"  {final_rank:<6}  {score:<12.6f}  {retrievers:<25}  {chunk}{star}")


# ---------------------------------------------------------------------------
# STEP 5 — Key takeaways
# ---------------------------------------------------------------------------

print()
print("--- STEP 5: Why RRF beats picking one retriever ---")
print()
print("  1. Agreement bonus")
print("     'task_service.py::create_task' was rank 2 (semantic) AND rank 4 (lexical).")
print("     RRF added both scores → it climbed higher than either rank alone.")
print()
print("  2. Covers blind spots")
print("     'check_task_limit' — lexical rank 1, absent from semantic.")
print("     'BillingService'   — semantic rank 1, absent from lexical.")
print("     Both appear in the fused list. Neither retriever's miss is final.")
print()
print("  3. No re-training needed")
print("     RRF is score-free — it only uses rank positions.")
print("     You can fuse any two retrievers without calibrating score scales.")
print()
print("  4. The k constant prevents over-rewarding rank 1")
print("     Without k: rank 1 = 1/1 = 1.0,  rank 2 = 1/2 = 0.5  (huge gap)")
print(f"     With  k={RRF_K}: rank 1 = 1/61 = 0.0164, rank 2 = 1/62 = 0.0161  (smooth)")
