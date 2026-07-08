"""
Multi-tenancy in a RAG system.

Scenario: "DevBot" is deployed org-wide as a shared incident-response
copilot. Every engineering team's postmortems and runbooks get indexed into
it -- but a checkout-team engineer asking "what caused our last outage?"
should never get back the payments-platform team's incident writeup (root
cause, which vendor broke, what internal system was involved, etc.). Each
team is effectively a "tenant" of the shared DevBot instance. The exact same
problem shows up if DevBot is instead sold as a hosted SaaS product to
separate customer companies -- "tenant" just becomes "customer org" instead
of "internal team".

Two common designs:

  (A) Shared collection + tenant_id metadata filter.
      One index for everyone. Cheap to run, easy to scale to many small
      tenants. But isolation is *soft*: it only holds if every single query
      remembers to pass `where={"tenant_id": ...}`. Forget it once, in one
      code path, and one team's retriever can surface another team's data.

  (B) Collection-per-tenant.
      Isolation is *structural*: the checkout team's client object literally
      cannot query the payments-platform collection, so there's no filter to
      forget. Stronger security boundary, but the number of
      collections/indexes grows with the number of tenants -- more moving
      parts to provision and manage (see 03_sharding_scaling.py for how this
      interacts with scaling).

This script demonstrates the leak in (A), the fix in (A), and then (B).
"""

import chromadb

CHECKOUT_DOCS = [
    ("checkout-1", "Postmortem: the Nov 2025 checkout outage was caused by a race condition in the cart-lock Redis key TTL.",
     {"tenant_id": "checkout", "confidential": True}),
    ("checkout-2", "Runbook: redeploy the checkout service with `make deploy SERVICE=checkout ENV=prod`.",
     {"tenant_id": "checkout", "confidential": False}),
    ("checkout-3", "The checkout team's on-call rotation is 'checkout-primary' in PagerDuty.",
     {"tenant_id": "checkout", "confidential": False}),
]

PAYMENTS_DOCS = [
    ("payments-1", "Postmortem: the Dec 2025 payments outage traced back to a stale feature flag disabling retries on the ledger writer.",
     {"tenant_id": "payments-platform", "confidential": True}),
    ("payments-2", "Runbook: roll back payments-platform with `kubectl rollout undo deployment/payments-platform -n prod`.",
     {"tenant_id": "payments-platform", "confidential": False}),
    ("payments-3", "The payments-platform on-call rotation is 'payments-primary' in PagerDuty.",
     {"tenant_id": "payments-platform", "confidential": False}),
]


def show(title: str, results: dict) -> None:
    print(f"\n--- {title} ---")
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        print(f"  [tenant={meta['tenant_id']}] {doc}")


def demo_shared_collection() -> None:
    print("\n===== Design A: shared collection + tenant_id filter =====")
    client = chromadb.Client()
    collection = client.get_or_create_collection("devbot_incidents_shared")
    all_docs = CHECKOUT_DOCS + PAYMENTS_DOCS
    collection.add(
        ids=[d[0] for d in all_docs],
        documents=[d[1] for d in all_docs],
        metadatas=[d[2] for d in all_docs],
    )

    query = "What caused our last production outage?"

    # bug: the checkout team's on-call handler forgot to scope the query.
    # This is the classic multi-tenant RAG mistake — it doesn't error,
    # it just quietly leaks payments-platform's incident root cause to a
    # checkout engineer.
    show(
        "LEAK -- query with no tenant filter (bug: forgot 'where')",
        collection.query(query_texts=[query], n_results=2),
    )

    # fix: every query path must filter by the requesting team/tenant.
    show(
        "FIXED -- same query, scoped with where={'tenant_id': 'checkout'}",
        collection.query(
            query_texts=[query], n_results=2,
            where={"tenant_id": "checkout"},
        ),
    )


def demo_collection_per_tenant() -> None:
    print("\n===== Design B: one collection per tenant =====")
    client = chromadb.Client()

    tenant_collections = {}
    for tenant, docs in [("checkout", CHECKOUT_DOCS), ("payments-platform", PAYMENTS_DOCS)]:
        coll = client.get_or_create_collection(f"devbot_incidents__{tenant}")
        coll.add(
            ids=[d[0] for d in docs],
            documents=[d[1] for d in docs],
            metadatas=[d[2] for d in docs],
        )
        tenant_collections[tenant] = coll

    # The checkout team's request handler only ever holds a handle to
    # checkout's collection. There is no `where` clause to forget --
    # payments-platform's data isn't reachable from this object at all.
    checkout_collection = tenant_collections["checkout"]
    show(
        "Checkout's own collection object -- payments-platform docs aren't even in scope",
        checkout_collection.query(
            query_texts=["What caused our last production outage?"], n_results=2
        ),
    )


if __name__ == "__main__":
    demo_shared_collection()
    demo_collection_per_tenant()

    print(
        "\nTakeaway: (A) shared collection is cheap and scales well to many "
        "small tenants, but isolation is only as strong as your least "
        "careful query path. (B) collection-per-tenant makes leaks "
        "structurally impossible, at the cost of managing N collections. "
        "Real systems often mix both: shard big/sensitive tenants into "
        "their own collection, and pool small tenants behind a metadata "
        "filter -- which is exactly the sharding trade-off in "
        "03_sharding_scaling.py."
    )
