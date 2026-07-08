"""
Metadata filtering in a vector DB (Chroma).

Scenario: "DevBot" is an internal RAG assistant indexed over an engineering
org's knowledge base -- runbooks, architecture decision records (ADRs),
postmortems, and API references contributed by different teams.

A vector search on its own only understands embeddings — it has no concept of
"only look at SRE runbooks" or "only docs from 2025 or later". Metadata
filtering fixes that: every chunk is stored with structured fields (team,
doc_type, confidentiality, year, ...) alongside its embedding, and the query
passes a `where` clause that Chroma applies as a hard pre-filter *before* the
nearest-neighbor search runs. This is what makes access control, freshness
windows, and multi-tenancy possible in a shared
index.
"""

import chromadb

DOCS = [
    ("d1", "To roll back a bad deploy, run `kubectl rollout undo deployment/<name> -n prod`.",
     {"team": "sre", "doc_type": "runbook", "confidentiality": "public", "year": 2024}),
    ("d2", "Restarting the ingest-worker pod clears most stuck-queue incidents; check Grafana first.",
     {"team": "sre", "doc_type": "runbook", "confidentiality": "public", "year": 2025}),
    ("d3", "Postmortem: the March 2025 checkout outage was caused by a missing circuit breaker on the pricing-service call.",
     {"team": "payments", "doc_type": "postmortem", "confidentiality": "internal", "year": 2025}),
    ("d4", "ADR-014: we chose Postgres over DynamoDB for the payments ledger for strong consistency guarantees.",
     {"team": "payments", "doc_type": "adr", "confidentiality": "internal", "year": 2023}),
    ("d5", "The /v2/payments endpoint requires an Idempotency-Key header on all POST requests.",
     {"team": "payments", "doc_type": "api-reference", "confidentiality": "public", "year": 2024}),
    ("d6", "Postmortem: a bad feature-flag rollout disabled retries on the ledger writer, dropping payments for 12 minutes.",
     {"team": "payments", "doc_type": "postmortem", "confidentiality": "internal", "year": 2025}),
    ("d7", "ADR-021: the mobile app moved from REST polling to a WebSocket feed for order-status updates.",
     {"team": "mobile", "doc_type": "adr", "confidentiality": "internal", "year": 2025}),
    ("d8", "The mobile crash-reporting pipeline uploads symbol files to Sentry on every release build.",
     {"team": "mobile", "doc_type": "runbook", "confidentiality": "public", "year": 2024}),
    ("d9", "The data-infra nightly ETL job re-indexes the analytics warehouse from the event stream.",
     {"team": "data-infra", "doc_type": "runbook", "confidentiality": "public", "year": 2024}),
    ("d10", "ADR-009: we adopted dbt for warehouse transformations instead of hand-rolled SQL scripts.",
     {"team": "data-infra", "doc_type": "adr", "confidentiality": "internal", "year": 2023}),
]


def build_collection() -> "chromadb.Collection":
    client = chromadb.Client()
    collection = client.get_or_create_collection("devbot_kb")
    collection.add(
        ids=[d[0] for d in DOCS],
        documents=[d[1] for d in DOCS],
        metadatas=[d[2] for d in DOCS],
    )
    return collection


def show(title: str, results: dict) -> None:
    print(f"\n--- {title} ---")
    for doc, meta, dist in zip(
        results["documents"][0], results["metadatas"][0], results["distances"][0]
    ):
        print(f"  [{meta['team']:<10}|{meta['doc_type']:<13}|{meta['confidentiality']:<8}"
              f"|{meta['year']}] (dist={dist:.3f}) {doc}")


if __name__ == "__main__":
    collection = build_collection()
    query = "How do I roll back a bad deployment?"

    # 1. No filter: pure semantic search over the whole knowledge base.
    #    Notice docs from unrelated teams (mobile, data-infra) can sneak
    #    into the results just by embedding similarity, even though the
    #    question is really an SRE/on-call question.
    show(
        "No metadata filter (search everything)",
        collection.query(query_texts=[query], n_results=4),
    )

    # 2. Simple equality filter: only look inside one team's docs.
    show(
        "Filter: team == 'sre'",
        collection.query(
            query_texts=[query], n_results=4,
            where={"team": "sre"},
        ),
    )

    # 3. Compound filter with $and: only internal payments postmortems
    #    (e.g. for an incident-review tool that should never surface
    #    public API docs when someone asks "what went wrong").
    show(
        "Filter: team == 'payments' AND doc_type == 'postmortem'",
        collection.query(
            query_texts=["what caused our last production incident?"], n_results=4,
            where={"$and": [
                {"team": "payments"},
                {"doc_type": "postmortem"},
            ]},
        ),
    )

    # 4. $in filter: search across a set of teams at once (e.g. an
    #    on-call engineer covering both sre and payments this week).
    show(
        "Filter: team in ['sre', 'payments']",
        collection.query(
            query_texts=["how does the deploy pipeline work"], n_results=4,
            where={"team": {"$in": ["sre", "payments"]}},
        ),
    )

    # 5. Range filter: only recent decisions/incidents (freshness window).
    show(
        "Filter: year >= 2025",
        collection.query(
            query_texts=[query], n_results=4,
            where={"year": {"$gte": 2025}},
        ),
    )

    print(
        "\nTakeaway: metadata filtering narrows the candidate set *before* "
        "vector similarity is scored. Same embeddings, same index — but the "
        "'where' clause enforces access control, recency, and topic scoping "
        "that pure semantic similarity can't guarantee on its own."
    )
