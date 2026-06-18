import argparse
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
from langchain_core.documents import Document
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from rank_bm25 import BM25Okapi
from langchain.agents.middleware import ModelCallLimitMiddleware
from langchain.agents.middleware import ToolCallLimitMiddleware



# ---------------------------------------------------------------------------
# 1. LOAD
# ---------------------------------------------------------------------------

CHUNK_SIZE = 1500
def load_codebase(repo_path: str) -> list:
   docs = []
   for path in Path(repo_path).rglob("*.py"):
       text = path.read_text(encoding="utf-8", errors="ignore")
       docs.append(Document(page_content=text, metadata={"source": str(path)}))
   return docs

# ---------------------------------------------------------------------------
# 2. CHUNK
# ---------------------------------------------------------------------------


def chunk_code(docs: list) -> list:
   splitter = RecursiveCharacterTextSplitter.from_language(
       language=Language.PYTHON,
       chunk_size=CHUNK_SIZE,
       chunk_overlap=32,
   )
   return splitter.split_documents(docs)



# ---------------------------------------------------------------------------
# 3. BUILD INDEXES
# ---------------------------------------------------------------------------

def build_vector_store(chunks: list) -> Chroma:
   embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
   return Chroma.from_documents(chunks, embedding=embeddings)

def build_bm25_index(chunks: list) -> tuple:
   tokenized = [doc.page_content.lower().split() for doc in chunks]
   return BM25Okapi(tokenized), chunks



# ---------------------------------------------------------------------------
# 4. RETRIEVAL TOOL — inner router agent with two sub-tools
#
# Architecture:
#   retrieval_tool (outer @tool)
#       └─ router agent (create_agent)
#               ├─ semantic_retrieval  — dense vector search via Chroma
#               └─ lexical_retrieval   — sparse BM25 keyword search
#
# The router agent receives the user query and decides which sub-tool(s)
# to call. The outer agent sees only retrieval_tool and delegates all
# retrieval decisions to it.
# ---------------------------------------------------------------------------


def build_retrieval_tool(vector_store: Chroma, bm25: BM25Okapi, bm25_docs: list):
   llm = init_chat_model("gpt-4o-mini", model_provider="openai", temperature=0)

   @tool
   def semantic_retrieval(query: str) -> str:
       """Retrieve code chunks using dense vector (semantic) similarity.
       Use for conceptual or intent-based queries about how something works."""
       docs = vector_store.similarity_search(query, k=4)
       print(f"  [semantic_retrieval] '{query}' → {len(docs)} chunks")
       return "\n\n".join(
           f"# {d.metadata.get('source', 'unknown')}\n{d.page_content}" for d in docs
       )

   @tool
   def lexical_retrieval(query: str) -> str:
       """Retrieve code chunks using BM25 keyword matching.
       Use for exact identifiers: function names, class names, error strings."""
       tokens = query.lower().split()
       scores = bm25.get_scores(tokens)
       top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:4]
       docs = [bm25_docs[i] for i in top_indices]
       print(f"  [lexical_retrieval]  '{query}' → {len(docs)} chunks")
       return "\n\n".join(
           f"# {d.metadata.get('source', 'unknown')}\n{d.page_content}" for d in docs
       )

   router_agent = create_agent(
       model=llm,
       tools=[semantic_retrieval, lexical_retrieval],
       system_prompt=(
           "You are a retrieval router for a Python codebase. "
           "Given a query, call the right retrieval tool:\n"
           "- semantic_retrieval: conceptual queries ('how does X work?', 'where is Y handled?')\n"
           "- lexical_retrieval: exact names or strings ('find ClassName', 'where is func_name called?')\n"
           "Return all retrieved content without summarising it."
       ),
   )


   @tool
   def retrieval_tool(query: str) -> str:
       """Retrieve relevant code from the codebase for any query.
       Internally routes to semantic or lexical retrieval as appropriate."""
       print(f"\n[retrieval_tool] Routing query: '{query}'")
       result = router_agent.invoke(
           {"messages": [{"role": "user", "content": query}]}
       )
       return str(result["messages"][-1].content)


   return retrieval_tool



# ---------------------------------------------------------------------------
# 5. OUTER AGENT
# ---------------------------------------------------------------------------


def build_agent(retrieval_tool):
   llm = init_chat_model("gpt-4o-mini", model_provider="openai", temperature=0)
   return create_agent(
       model=llm,
       tools=[retrieval_tool],
       system_prompt=(
           "You are a senior engineer. Always use retrieval_tool before answering. "
           "Reference specific file and function names. "
           "If not found say 'I could not find that in the codebase'."
       ),
       middleware=[
           ModelCallLimitMiddleware(run_limit=5, exit_behavior="end"),
           ToolCallLimitMiddleware(tool_name="retriever_tool", run_limit=2, exit_behavior="end")
       ]
   )



# ---------------------------------------------------------------------------
# 6. MAIN
# ---------------------------------------------------------------------------


if __name__ == "__main__":
   parser = argparse.ArgumentParser()
   parser.add_argument("--repo", default=str(Path(__file__).parent.parent / "sample_project"))
   args = parser.parse_args()
   repo_path = str(Path(args.repo).resolve())


   docs = load_codebase(repo_path)
   chunks = chunk_code(docs)
   print(f"Loaded {len(docs)} files → {len(chunks)} chunks (chunk_size={CHUNK_SIZE})")


   vector_store = build_vector_store(chunks)
   bm25, bm25_docs = build_bm25_index(chunks)


   retrieval_tool = build_retrieval_tool(vector_store, bm25, bm25_docs)
   agent = build_agent(retrieval_tool)


   print("Ready. Ask your question. Type 'exit' to quit")
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










