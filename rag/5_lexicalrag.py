import argparse
from pathlib import Path
from dotenv import load_dotenv


load_dotenv()


from langchain_core.documents import Document
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from rank_bm25 import BM25Okapi
from langchain_openai import ChatOpenAI
from langchain_core.tools.retriever import create_retriever_tool
from langchain.agents import create_agent




# ---------------------------------------------------------------------------
# 1. LOAD
# ---------------------------------------------------------------------------


CHUNK_SIZE = 256


def load_codebase(repo_path: str) -> list:
   docs = []
   for path in Path(repo_path).rglob("*.py"):
       text = path.read_text(encoding="utf-8", errors="ignore")
       docs.append(Document(page_content=text, metadata={"source": str(path)}))
   return docs




# ---------------------------------------------------------------------------
# 2. CHUNK — same character-based splits as demo 1
# ---------------------------------------------------------------------------


def chunk_code(docs: list) -> list:
   splitter = RecursiveCharacterTextSplitter.from_language(
       language=Language.PYTHON,
       chunk_size=CHUNK_SIZE,
       chunk_overlap=32,
   )
   return splitter.split_documents(docs)




# ---------------------------------------------------------------------------
# 3. BM25 INDEX — no embeddings, pure term frequency
# ---------------------------------------------------------------------------


class BM25Retriever(BaseRetriever):
   """Thin LangChain-compatible retriever backed by rank_bm25."""
   docs: list
   bm25: object
   k: int = 4


   class Config:
       arbitrary_types_allowed = True


   def _get_relevant_documents(
       self, query: str, *, run_manager: CallbackManagerForRetrieverRun
   ) -> list:
       tokens = query.lower().split()
       scores = self.bm25.get_scores(tokens)
       top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[: self.k]
       return [self.docs[i] for i in top_indices]




def build_bm25_retriever(chunks: list) -> BM25Retriever:
   tokenized = [doc.page_content.lower().split() for doc in chunks]
   bm25 = BM25Okapi(tokenized)
   return BM25Retriever(docs=chunks, bm25=bm25)




# ---------------------------------------------------------------------------
# 4. RETRIEVER + AGENT
# ---------------------------------------------------------------------------


def build_agent(retriever: BaseRetriever):
   retriever_tool = create_retriever_tool(
       retriever,
       name="search_codebase",
       description="Search the codebase for relevant functions, classes, or logic.",
   )
   llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
   return create_agent(
       llm, tools=[retriever_tool],
       system_prompt=(
           "You are a senior engineer. Always use search_codebase before answering. "
           "Reference specific file and function names. "
           "If not found say 'I could not find that in the codebase'."
       ),
   )




# ---------------------------------------------------------------------------
# 5. MAIN
# ---------------------------------------------------------------------------


if __name__ == "__main__":
   parser = argparse.ArgumentParser()
   parser.add_argument("--repo", default=str(Path(__file__).parent.parent / "sample_project"))
   args = parser.parse_args()
   repo_path = str(Path(args.repo).resolve())






   docs = load_codebase(repo_path)
   chunks = chunk_code(docs)
   print(f"Loaded {len(docs)} files → {len(chunks)} chunks (chunk_size={CHUNK_SIZE})")




   retriever = build_bm25_retriever(chunks)
   agent = build_agent(retriever)
  


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
