"""
RAPTOR RAG (manual implementation) with latest LangChain APIs.

RAPTOR = Recursive Abstractive Processing for Tree-Organized Retrieval
Paper: https://arxiv.org/abs/2401.18059

Core idea:
1) Build leaf chunks from documents.
2) Recursively cluster semantically similar chunks.
3) Summarize each cluster into a parent node.
4) Repeat to form a hierarchy (tree of abstractions).
5) At query time, retrieve from ALL levels (leaf + summaries), not just raw chunks.

This script implements that end-to-end using:
- `init_chat_model` + `create_agent` (latest LangChain APIs)
- `init_embeddings` + `InMemoryVectorStore`
- Plain Python for clustering/tree building (no RAPTOR library).
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass

from agentic_ai.config import OPENAI_MODEL

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.embeddings import init_embeddings
from langchain.tools import tool
from langchain_core.documents import Document
from langchain_core.messages import ToolMessage
from langchain_core.vectorstores import InMemoryVectorStore

EMBEDDING_MODEL = "text-embedding-3-small"
MAX_TREE_LEVELS = 3
MAX_CLUSTER_SIZE = 3
SIMILARITY_THRESHOLD = 0.55
RETRIEVE_K = 4


@dataclass
class TreeNode:
    """One node in the RAPTOR tree."""

    node_id: str
    level: int
    text: str
    children_ids: list[str]


CORPUS_DOCS = [
    (
        "refund_policy",
        "Acme refunds are available for regular products within 30 days of purchase. "
        "Customers must submit the request through support.acme.com. "
        "Digital products are refundable only when proven defective at delivery.",
    ),
    (
        "refund_timing",
        "Approved refunds are processed in 5 to 7 business days. "
        "Customers receive an email confirmation when refund initiation begins. "
        "Bank settlement times may vary by payment method.",
    ),
    (
        "shipping_policy",
        "Express shipping costs $15 and usually arrives within 2 business days. "
        "Standard shipping is free above $50 and typically arrives in 5 to 7 business days.",
    ),
    (
        "security_policy",
        "Passwords must include at least 12 characters, uppercase letters, numbers, and symbols. "
        "Password reset links expire after 30 minutes. Passwords expire every 90 days.",
    ),
    (
        "account_lockout",
        "After 5 failed login attempts, accounts are temporarily locked for 15 minutes. "
        "Security alerts are sent to the registered email for suspicious access attempts.",
    ),
]


def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """Cosine similarity in plain Python."""
    dot = sum(a * b for a, b in zip(vec_a, vec_b, strict=True))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def sentence_chunks(text: str, max_sentences: int = 2) -> list[str]:
    """
    Small chunker for demo purposes.
    RAPTOR starts from leaf chunks, so this creates initial leaves.
    """
    parts = [s.strip() for s in text.split(".") if s.strip()]
    chunks: list[str] = []
    bucket: list[str] = []
    for part in parts:
        bucket.append(part + ".")
        if len(bucket) >= max_sentences:
            chunks.append(" ".join(bucket))
            bucket = []
    if bucket:
        chunks.append(" ".join(bucket))
    return chunks


def build_leaf_nodes() -> list[TreeNode]:
    """Create level-0 nodes from corpus chunks."""
    leaves: list[TreeNode] = []
    counter = 0
    for source_id, text in CORPUS_DOCS:
        for chunk in sentence_chunks(text, max_sentences=2):
            leaves.append(
                TreeNode(
                    node_id=f"L0_{counter}",
                    level=0,
                    text=f"[{source_id}] {chunk}",
                    children_ids=[],
                )
            )
            counter += 1
    return leaves


def greedy_semantic_clusters(
    texts: list[str],
    embeddings_model,
    threshold: float = SIMILARITY_THRESHOLD,
    max_cluster_size: int = MAX_CLUSTER_SIZE,
) -> list[list[int]]:
    """
    Greedy clustering by semantic similarity.

    Why this exists:
    RAPTOR paper clusters chunks before summarizing.
    Here we avoid extra dependencies and implement a simple, readable variant.
    """
    vectors = embeddings_model.embed_documents(texts)
    clusters: list[list[int]] = []

    for idx, vec in enumerate(vectors):
        best_cluster = -1
        best_score = -1.0

        # Find the most similar existing cluster centroid.
        for c_idx, cluster in enumerate(clusters):
            if len(cluster) >= max_cluster_size:
                continue
            centroid = [
                sum(vectors[i][dim] for i in cluster) / len(cluster)
                for dim in range(len(vec))
            ]
            score = cosine_similarity(vec, centroid)
            if score > best_score:
                best_score = score
                best_cluster = c_idx

        # If no cluster is similar enough, create a new one.
        if best_cluster == -1 or best_score < threshold:
            clusters.append([idx])
        else:
            clusters[best_cluster].append(idx)

    return clusters


def build_summarizer_agent():
    """Agent that produces concise cluster summaries (parent nodes)."""
    return create_agent(
        model=init_chat_model(f"openai:{OPENAI_MODEL}", temperature=0),
        tools=[],
        system_prompt=(
            "You summarize related policy chunks into one compact paragraph.\n"
            "Keep concrete facts (numbers, limits, conditions) and remove repetition.\n"
            "Output plain text only."
        ),
    )


def summarize_cluster(agent, cluster_texts: list[str]) -> str:
    """Summarize one semantic cluster into a parent node text."""
    payload = "\n\n".join(f"- {t}" for t in cluster_texts)
    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": f"Summarize this cluster for retrieval:\n\n{payload}",
                }
            ]
        }
    )
    return str(result["messages"][-1].content).strip()


def build_raptor_tree(embeddings_model) -> list[TreeNode]:
    """
    Build RAPTOR hierarchy bottom-up:
    leaves -> cluster+summary -> higher-level summaries.
    """
    summarizer = build_summarizer_agent()
    all_nodes: list[TreeNode] = []

    current_level_nodes = build_leaf_nodes()
    all_nodes.extend(current_level_nodes)
    next_node_counter = 0

    for level in range(1, MAX_TREE_LEVELS + 1):
        if len(current_level_nodes) <= 1:
            break

        texts = [node.text for node in current_level_nodes]
        clusters = greedy_semantic_clusters(texts, embeddings_model)

        # If clustering can't merge anything, stop recursion.
        if len(clusters) == len(current_level_nodes):
            break

        parent_nodes: list[TreeNode] = []
        for cluster in clusters:
            child_nodes = [current_level_nodes[i] for i in cluster]
            summary = summarize_cluster(summarizer, [n.text for n in child_nodes])
            parent_nodes.append(
                TreeNode(
                    node_id=f"L{level}_{next_node_counter}",
                    level=level,
                    text=summary,
                    children_ids=[n.node_id for n in child_nodes],
                )
            )
            next_node_counter += 1

        all_nodes.extend(parent_nodes)
        current_level_nodes = parent_nodes

    return all_nodes


def build_vectorstores(nodes: list[TreeNode], embeddings_model):
    """
    Build two stores:
    - baseline_store: only leaves (traditional chunk retrieval)
    - raptor_store: leaves + summary nodes (RAPTOR retrieval)
    """
    leaves = [n for n in nodes if n.level == 0]
    leaf_docs = [
        Document(page_content=n.text, metadata={"node_id": n.node_id, "level": n.level})
        for n in leaves
    ]
    all_docs = [
        Document(page_content=n.text, metadata={"node_id": n.node_id, "level": n.level})
        for n in nodes
    ]
    baseline_store = InMemoryVectorStore.from_documents(leaf_docs, embeddings_model)
    raptor_store = InMemoryVectorStore.from_documents(all_docs, embeddings_model)
    return baseline_store, raptor_store


def build_qa_agent(raptor_store: InMemoryVectorStore):
    """QA agent that answers from RAPTOR retrieval results."""

    @tool(response_format="content_and_artifact")
    def raptor_search(query: str):
        """Retrieve context from RAPTOR index (multi-level tree nodes)."""
        docs = raptor_store.similarity_search(query, k=RETRIEVE_K)
        content = "\n\n".join(doc.page_content for doc in docs)
        artifact = [
            {
                "text": doc.page_content,
                "node_id": doc.metadata.get("node_id"),
                "level": doc.metadata.get("level"),
            }
            for doc in docs
        ]
        return content, artifact

    return create_agent(
        model=init_chat_model(f"openai:{OPENAI_MODEL}", temperature=0),
        tools=[raptor_search],
        system_prompt=(
            "You are a policy assistant.\n"
            "Always call raptor_search first.\n"
            "Answer only from retrieved content and keep answers concise."
        ),
    )


def print_tree_summary(nodes: list[TreeNode]) -> None:
    """Show how many nodes were created per level."""
    print("\nRAPTOR tree stats:")
    max_level = max(node.level for node in nodes)
    for level in range(max_level + 1):
        count = sum(1 for n in nodes if n.level == level)
        print(f"  level {level}: {count} nodes")


def compare_retrieval(question: str, baseline_store, raptor_store) -> None:
    """Print baseline vs RAPTOR retrieval results for learning."""
    baseline = baseline_store.similarity_search(question, k=RETRIEVE_K)
    raptor = raptor_store.similarity_search(question, k=RETRIEVE_K)

    print("\n" + "=" * 90)
    print(f"Question: {question}")
    print("=" * 90)

    print("\n[Baseline RAG] leaf-only retrieval:")
    for i, doc in enumerate(baseline, start=1):
        print(f"  {i}. (level {doc.metadata.get('level')}) {doc.page_content}")

    print("\n[RAPTOR RAG] retrieval across tree levels:")
    for i, doc in enumerate(raptor, start=1):
        print(f"  {i}. (level {doc.metadata.get('level')}) {doc.page_content}")


def run_qa(agent, question: str) -> None:
    """Run final QA and print retrieved levels used by the tool."""
    result = agent.invoke({"messages": [{"role": "user", "content": question}]})
    answer = str(result["messages"][-1].content).strip()

    retrieved_levels: list[int] = []
    for msg in result["messages"]:
        if isinstance(msg, ToolMessage):
            artifact = getattr(msg, "artifact", None)
            if isinstance(artifact, list):
                for item in artifact:
                    lvl = item.get("level")
                    if isinstance(lvl, int):
                        retrieved_levels.append(lvl)

    print("\n[Final Answer]")
    print(answer)
    if retrieved_levels:
        print(f"\nRetrieved levels used: {retrieved_levels}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Manual RAPTOR RAG with recursive cluster summaries."
    )
    parser.add_argument(
        "--question",
        type=str,
        default="What is the policy for digital refunds and how long do approved refunds take?",
        help="Question to ask the RAPTOR pipeline.",
    )
    args = parser.parse_args()

    embeddings_model = init_embeddings(f"openai:{EMBEDDING_MODEL}")
    nodes = build_raptor_tree(embeddings_model)
    print_tree_summary(nodes)

    baseline_store, raptor_store = build_vectorstores(nodes, embeddings_model)
    compare_retrieval(args.question, baseline_store, raptor_store)

    qa_agent = build_qa_agent(raptor_store)
    run_qa(qa_agent, args.question)


if __name__ == "__main__":
    main()
