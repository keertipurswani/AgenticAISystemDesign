"""
PageIndex RAG demo — vectorless retrieval via agentic tree navigation.

Based on: https://github.com/VectifyAI/PageIndex

How PageIndex works:
  INDEX TIME  — build a hierarchical tree index (title + summary per node).
                No embeddings. No vector store.

  QUERY TIME  — an agent reasons over the tree using three tools:
                  get_structure()          → full tree (summaries only, no source)
                  get_chunk_content(id)    → actual source of a specific chunk
                The agent drills into relevant files and fetches only the chunks
                it needs. No cosine search.

Tree structure (3 levels):
  directory  (branch) — LLM summary
  └─ file    (branch) — LLM summary
     └─ chunk (leaf)  — first line as summary, source as content

Reference: VectifyAI/PageIndex — "similarity ≠ relevance, retrieval requires reasoning"
"""

import argparse
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from langchain.agents import create_agent
from langchain.agents.middleware import ModelCallLimitMiddleware, ToolCallLimitMiddleware
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_core.documents import Document
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter

LLM_MODEL = "gpt-4o-mini"
CHUNK_SIZE = 1500


# ---------------------------------------------------------------------------
# 1. PAGE INDEX TREE
# ---------------------------------------------------------------------------

def _llm_summary(llm, prompt: str) -> str:
    return llm.invoke(prompt).content.strip()


def _build_chunk_nodes(file_node_id: str, source: str) -> list[dict]:
    # Split the file into chunks using Python-aware separators (class/def boundaries first).
    # Each chunk becomes a leaf — the agent fetches these by node_id at query time.
    splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.PYTHON, chunk_size=CHUNK_SIZE, chunk_overlap=32
    )
    chunks = splitter.split_text(source)
    return [
        {
            "title": f"chunk_{i + 1}",
            "node_id": f"{file_node_id}::chunk_{i + 1}",
            "summary": chunk.splitlines()[0][:120],  # first line as a lightweight preview
            "is_leaf": True,
            "content": chunk,   # actual source — only exposed when agent calls get_chunk_content
            "nodes": [],
        }
        for i, chunk in enumerate(chunks)
    ]


def _build_tree(path: Path, base: Path, llm) -> dict | None:
    rel = str(path.relative_to(base)).replace("\\", "/")

    if path.is_file() and path.suffix == ".py":
        source = path.read_text(encoding="utf-8", errors="ignore")
        print(f"  Summarising {rel} ...")

        # LLM reads the file and writes a summary stored in the index.
        # The agent sees this summary (not the source) when browsing get_structure().

        summary = _llm_summary(
            llm,
            f"In 1-2 sentences, summarise this Python file for a search index.\n"
            f"File: {rel}\n\n```python\n{source[:3000]}\n```",
        )

        # File is a branch node — its children are the source chunks.
        chunks = _build_chunk_nodes(rel, source)
        return {
            "title": path.name,
            "node_id": rel,
            "summary": summary,
            "is_leaf": False,
            "nodes": chunks,
        }

    if path.is_dir():
        # Recurse into subdirectories; skip hidden dirs and __pycache__.
        children = [
            node
            for child in sorted(path.iterdir())
            if not child.name.startswith(".") and child.name != "__pycache__"
            if (node := _build_tree(child, base, llm)) is not None
        ]
        if not children:
            return None
        # Directory summary is derived from the names of its children (no source needed).
        summary = _llm_summary(
            llm,
            f"In 1 sentence, summarise this Python package for a search index.\n"
            f"Package: {path.name}\nContains: {', '.join(c['title'] for c in children)}",
        )
        return {
            "title": path.name,
            "node_id": rel or "root",
            "summary": summary,
            "is_leaf": False,
            "nodes": children,
        }

    return None


def build_page_index(repo_path: str) -> dict:
    # Entry point: walk the repo and build the full tree. One LLM call per file/dir.
    base = Path(repo_path)
    llm = init_chat_model(LLM_MODEL, model_provider="openai", temperature=0)
    root = _build_tree(base, base, llm) or {
        "title": base.name, "node_id": "root", "summary": "Empty project",
        "is_leaf": False, "nodes": [],
    }
    root["node_id"] = "root"
    root["title"] = base.name
    return root


def _count_leaves(node: dict) -> int:
    if node["is_leaf"]:
        return 1
    return sum(_count_leaves(c) for c in node.get("nodes", []))


def _find_node(node: dict, node_id: str) -> dict | None:
    # Depth-first search for a node by its ID.
    if node["node_id"] == node_id:
        return node
    for child in node.get("nodes", []):
        found = _find_node(child, node_id)
        if found:
            return found
    return None


def _tree_to_str(node: dict, indent: int = 0) -> str:
    # Renders the tree showing only node_id + summary — no source code.
    # This is what the agent reads via get_structure() to decide what to fetch.
    prefix = "  " * indent
    icon = "📄" if node["is_leaf"] else "📁"
    lines = [f"{prefix}{icon} [{node['node_id']}] {node['title']}",
             f"{prefix}   {node['summary']}"]
    for child in node.get("nodes", []):
        lines.append(_tree_to_str(child, indent + 1))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 2. PAGEINDEX AGENT
# ---------------------------------------------------------------------------

def build_pageindex_agent(index: dict):
    llm = init_chat_model(LLM_MODEL, model_provider="openai", temperature=0)

    @tool
    def get_structure() -> str:
        """Get the full codebase tree with summaries. Call this first to find relevant chunks."""
        return _tree_to_str(index)

    @tool
    def get_chunk_content(node_id: str) -> str:
        """Get the source code of a specific chunk by node_id (must be a leaf node)."""
        node = _find_node(index, node_id)
        if node is None:
            return f"node_id '{node_id}' not found. Call get_structure() for valid IDs."
        if not node["is_leaf"]:
            children = [c["node_id"] for c in node.get("nodes", [])]
            return f"'{node_id}' is a branch. Children: {children}"
        return node.get("content", "(empty)")

    return create_agent(
        model=llm,
        tools=[get_structure, get_chunk_content],
        system_prompt=(
            "You are a senior engineer using PageIndex to search a codebase.\n"
            "Always call get_structure() first to find relevant chunk node_ids.\n"
            "Then call get_chunk_content(node_id) only for relevant chunks.\n"
            "Answer from retrieved content only. Reference file and function names.\n"
            "If not found say 'I could not find that in the codebase'."
        ),
        middleware=[
            ModelCallLimitMiddleware(run_limit=8, exit_behavior="end"),
            ToolCallLimitMiddleware(tool_name="get_chunk_content", run_limit=4, exit_behavior="end"),
        ],
    )


# ---------------------------------------------------------------------------
# 3. MAIN
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="PageIndex RAG: agentic tree navigation, no vectors."
    )
    parser.add_argument(
        "--repo",
        default=str(Path(__file__).parent.parent / "sample_project"),
    )
    args = parser.parse_args()
    repo_path = str(Path(args.repo).resolve())


    print("Building PageIndex tree ...")
    index = build_page_index(repo_path)
    print(f"PageIndex ready: {_count_leaves(index)} chunks — no vectors, no embeddings\n")


    print(_tree_to_str(index))


    agent = build_pageindex_agent(index)
    

    print("\nReady. Ask your question. Type 'exit' to quit")
    while True:
        question = input("\nYou: ").strip()
        if not question or question.lower() in ("exit", "quit"):
            break

        for step in agent.stream(
            {"messages": [{"role": "user", "content": question}]},
            stream_mode="values",
        ):
            last_msg = step["messages"][-1]
            if not getattr(last_msg, "tool_calls", None):
                print(f"Agent: {last_msg.content}")
