import sys
import types
import argparse
from pathlib import Path
from dotenv import load_dotenv


load_dotenv()


# ragas 0.4.x imports ChatVertexAI from langchain_community.chat_models.vertexai,
# which was removed in langchain-community 0.4. Inject a shim before ragas loads.
if "langchain_community.chat_models.vertexai" not in sys.modules:
   from langchain_google_vertexai import ChatVertexAI as _ChatVertexAI
   _shim = types.ModuleType("langchain_community.chat_models.vertexai")
   _shim.ChatVertexAI = _ChatVertexAI
   sys.modules["langchain_community.chat_models.vertexai"] = _shim


from langchain_core.documents import Document
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from pydantic import ConfigDict
from langchain_classic.retrievers.ensemble import EnsembleRetriever
from rank_bm25 import BM25Okapi
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings as LangchainOpenAIEmbeddings
from langchain_core.tools.retriever import create_retriever_tool
from langchain.agents import create_agent
from langchain.agents.middleware import ModelCallLimitMiddleware
from langchain.agents.middleware import ToolCallLimitMiddleware


from ragas import EvaluationDataset, SingleTurnSample, evaluate
from ragas.metrics._context_precision import LLMContextPrecisionWithReference
from ragas.metrics._context_recall import LLMContextRecall
from ragas.metrics._faithfulness import Faithfulness
from ragas.metrics._answer_relevance import AnswerRelevancy
from ragas.metrics._factual_correctness import FactualCorrectness
from ragas.llms import llm_factory


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
# 2. CHUNK — character-based splits shared by both retrievers
# ---------------------------------------------------------------------------


def chunk_code(docs: list) -> list:
   splitter = RecursiveCharacterTextSplitter.from_language(
       language=Language.PYTHON,
       chunk_size=CHUNK_SIZE,
       chunk_overlap=32,
   )
   return splitter.split_documents(docs)


# ---------------------------------------------------------------------------
# 3a. SEMANTIC — dense vector retriever
# ---------------------------------------------------------------------------

def build_vector_retriever(chunks: list):
   embeddings = LangchainOpenAIEmbeddings(model="text-embedding-3-small")
   vector_store = Chroma.from_documents(chunks, embedding=embeddings)
   return vector_store.as_retriever(search_kwargs={"k": 2})



# ---------------------------------------------------------------------------
# 3b. LEXICAL — sparse BM25 retriever
# ---------------------------------------------------------------------------


class BM25Retriever(BaseRetriever):
   """Thin LangChain-compatible retriever backed by rank_bm25."""
   docs: list
   bm25: object
   k: int = 4

   model_config = ConfigDict(arbitrary_types_allowed=True)

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
# 3c. HYBRID — fuse both via Reciprocal Rank Fusion
# ---------------------------------------------------------------------------

def build_hybrid_retriever(chunks: list, weights: list = None):
   vector_retriever = build_vector_retriever(chunks)
   bm25_retriever = build_bm25_retriever(chunks)
   return EnsembleRetriever( #fusion algo
       retrievers=[vector_retriever, bm25_retriever],
       weights=weights or [0.5, 0.5],
   )



# ---------------------------------------------------------------------------
# 4. AGENT
# ---------------------------------------------------------------------------


def build_agent(retriever):
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
       middleware=[
           ModelCallLimitMiddleware(run_limit=5, exit_behavior="end"),
           ToolCallLimitMiddleware(tool_name="retriever_tool", run_limit=2, exit_behavior="end")
       ]
   )



def run_agent(agent, question: str) -> tuple[str, list[str]]:
   result = agent.invoke({"messages": [{"role": "user", "content": question}]})
   answer = result["messages"][-1].content


   contexts = []
   for msg in result["messages"]:
       if type(msg).__name__ == "ToolMessage" and isinstance(msg.content, str):
           contexts.append(msg.content)


   return answer, contexts





# ---------------------------------------------------------------------------
# Golden Dataset
# ---------------------------------------------------------------------------


from golden_dataset import TEST_CASES




# ---------------------------------------------------------------------------
# 5. MAIN
# ---------------------------------------------------------------------------


if __name__ == "__main__":
   parser = argparse.ArgumentParser()
   parser.add_argument("--repo", default=str(Path(__file__).parent.parent / "sample_project"))
   parser.add_argument(
       "--weights", nargs=2, type=float, default=[0.5, 0.5],
       metavar=("SEMANTIC", "LEXICAL"),
       help="Retriever blend weights (must sum to 1.0). Default: 0.5 0.5",
   )
   args = parser.parse_args()
   repo_path = str(Path(args.repo).resolve())


   docs = load_codebase(repo_path)
   chunks = chunk_code(docs)
   print(f"Loaded {len(docs)} files → {len(chunks)} chunks (chunk_size={CHUNK_SIZE})")
   print(f"Hybrid weights — semantic: {args.weights[0]}, lexical: {args.weights[1]}")


   retriever = build_hybrid_retriever(chunks, weights=args.weights)
   agent = build_agent(retriever)


   # Collect RAG outputs
   print("\nRunning agent on test cases...")
   samples = []
   for tc in TEST_CASES:
       answer, contexts = run_agent(agent, tc["question"])
       samples.append(SingleTurnSample(
           user_input=tc["question"],
           response=answer,
           retrieved_contexts=contexts,
           reference=tc["reference"],
       ))


       print(f"\nQ: {tc['question']}")
       print(f"Contexts retrieved ({len(contexts)}):")
       for i, ctx in enumerate(contexts, 1):
           print(f"  [{i}] {ctx[:200]}{'...' if len(ctx) > 200 else ''}")
       print(f"Answer: {answer}")


   # RAGAS evaluation
   print("\nRunning RAGAS evaluation (GPT-4o as judge)...")
   from openai import OpenAI
   openai_client = OpenAI()
   evaluator_llm = llm_factory("gpt-4o", client=openai_client)
   lc_embeddings = LangchainOpenAIEmbeddings(model="text-embedding-3-small")
   eval_dataset = EvaluationDataset(samples=samples)


   results = evaluate(
       dataset=eval_dataset,
       metrics=[
           LLMContextPrecisionWithReference(),
           LLMContextRecall(),
           Faithfulness(),
           AnswerRelevancy(embeddings=lc_embeddings),
           FactualCorrectness(),
       ],
       llm=evaluator_llm,
   )


   # Scorecard
   df = results.to_pandas()
   metric_cols = [
       "llm_context_precision_with_reference",
       "context_recall",
       "faithfulness",
       "answer_relevancy",
       "factual_correctness",
   ]


   print("\nRAGAS SCORECARD — Hybrid RAG on sample_project")
   print("\nPer-question breakdown:")
   for _, row in df.iterrows():
       print(f"\n  Q: {row['user_input'][:65]}")
       for col in metric_cols:
           if col in row:
               val = row[col]
               icon = "✅" if val >= 0.7 else ("⚠️ " if val >= 0.5 else "❌")
               print(f"    {icon} {col:<42}: {val:.3f}")


   print("\nAggregate averages:")
   for col in metric_cols:
       if col in df.columns:
           avg = df[col].mean()
           icon = "✅" if avg >= 0.7 else ("⚠️ " if avg >= 0.5 else "❌")
           print(f"  {icon} {col:<42}: {avg:.3f}")
   print()














