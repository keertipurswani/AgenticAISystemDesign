"""
HyDE (Hypothetical Document Embeddings) demo using latest LangChain APIs.

References used:
- HyDE paper: "Precise Zero-Shot Dense Retrieval without Relevance Labels" (arXiv:2212.10496)
- LangChain HyDE docs: generate hypothetical answer text, then embed that text for retrieval.

This script intentionally implements HyDE manually (without langchain-classic HyDE retriever)
so you can understand each step clearly.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.embeddings import init_embeddings
from langchain.tools import tool
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter


EMBEDDING_MODEL = "text-embedding-3-small"
OPENAI_MODEL = "gpt-4o-mini"
CHUNK_SIZE = 1500
TOP_K = 4


# ---------------------------------------------------------------------------
# 1. LOAD
# ---------------------------------------------------------------------------

def load_codebase(repo_path: str) -> list[Document]:
    docs = []
    for path in Path(repo_path).rglob("*.py"):
        text = path.read_text(encoding="utf-8", errors="ignore")
        docs.append(Document(page_content=text, metadata={"source": str(path)}))
    return docs


# ---------------------------------------------------------------------------
# 2. CHUNK
# ---------------------------------------------------------------------------

def chunk_code(docs: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.PYTHON,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=32,
    )
    return splitter.split_documents(docs)


# ---------------------------------------------------------------------------
# 3. EMBED & STORE
# ---------------------------------------------------------------------------

def build_vector_store(chunks: list[Document]) -> InMemoryVectorStore:
    embeddings = init_embeddings(f"openai:{EMBEDDING_MODEL}")
    return InMemoryVectorStore.from_documents(chunks, embeddings)


# ---------------------------------------------------------------------------
# 4. AGENTS
# ---------------------------------------------------------------------------

def build_hyde_generator():
    """
    Agent that writes a hypothetical answer-document from the user question.

    HyDE core idea:
    - Do NOT embed the raw question directly.
    - First generate a plausible answer passage (hypothetical document).
    - Then embed that hypothetical document and retrieve nearest real docs.
    """
    return create_agent(
        model=init_chat_model(f"openai:{OPENAI_MODEL}", temperature=0),
        tools=[],
        system_prompt=(
            "You generate a single hypothetical code snippet or document for retrieval.\n"
            "Write a compact, plausible-looking passage (4-7 sentences or a short code snippet) "
            "that could answer the question about a Python codebase.\n"
            "Do not add bullet points, no disclaimers, and no markdown."
        ),
    )


def build_qa_agent(vector_store: InMemoryVectorStore):
    """
    Final QA agent that answers only from retrieved context via HyDE retrieval.
    """
    hyde_agent = build_hyde_generator()

    @tool(response_format="content_and_artifact")
    def search_codebase(query: str):
        """Retrieve code chunks with HyDE: generate a hypothetical doc, then retrieve by its embedding."""
        hypo_response = hyde_agent.invoke(
            {"messages": [{"role": "user", "content": query}]}
        )
        hypothetical_doc = str(hypo_response["messages"][-1].content).strip()
        print(f"\n[HyDE] Hypothetical document:\n{hypothetical_doc}\n")
        docs = vector_store.similarity_search(hypothetical_doc, k=TOP_K)

        content = "\n\n".join(
            f"# {doc.metadata.get('source', 'unknown')}\n{doc.page_content}"
            for doc in docs
        )
        artifact = {
            "hypothetical_document": hypothetical_doc,
            "retrieved_docs": [doc.page_content for doc in docs],
        }
        return content, artifact

    qa_agent = create_agent(
        model=init_chat_model(f"openai:{OPENAI_MODEL}", temperature=0),
        tools=[search_codebase],
        system_prompt=(
            "You are a senior engineer. Always use search_codebase before answering.\n"
            "Reference specific file and function names.\n"
            "If not found say 'I could not find that in the codebase'."
        ),
    )
    return qa_agent, hyde_agent


# ---------------------------------------------------------------------------
# 5. MAIN
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HyDE RAG demo over a Python codebase.")
    parser.add_argument("--repo", default=str(Path(__file__).parent.parent / "sample_project"))
    args = parser.parse_args()
    repo_path = str(Path(args.repo).resolve())


    docs = load_codebase(repo_path)
    chunks = chunk_code(docs)
    print(f"Loaded {len(docs)} files → {len(chunks)} chunks (chunk_size={CHUNK_SIZE})")


    vector_store = build_vector_store(chunks)
    qa_agent, hyde_agent = build_qa_agent(vector_store)


    print("Ready. Ask your question. Type 'exit' to quit")
    while True:
        question = input("\nYou: ").strip()
        if not question or question.lower() in ("exit", "quit"):
            break

        for step in qa_agent.stream(
            {"messages": [{"role": "user", "content": question}]},
            stream_mode="values",
        ):
            last_msg = step["messages"][-1]
            if not getattr(last_msg, "tool_calls", None):
                print(f"Agent: {last_msg.content}")
