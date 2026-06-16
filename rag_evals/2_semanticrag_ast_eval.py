import sys
import ast
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
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings as LangchainOpenAIEmbeddings
from langchain_core.tools.retriever import create_retriever_tool
from langchain.agents import create_agent

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

def load_codebase(repo_path: str) -> list:
    docs = []
    for path in Path(repo_path).rglob("*.py"):
        text = path.read_text(encoding="utf-8", errors="ignore")
        docs.append(Document(page_content=text, metadata={"source": str(path)}))
    return docs


# ---------------------------------------------------------------------------
# 2. CHUNK — AST-based (one chunk per class or module-level function)
# ---------------------------------------------------------------------------

def chunk_code(docs: list) -> list:
    chunks = []

    for doc in docs:
        source = doc.metadata.get("source", "unknown")
        code = doc.page_content

        try:
            tree = ast.parse(code)
        except SyntaxError:
            chunks.append(doc)
            continue

        for node in ast.walk(tree):
            for child in ast.iter_child_nodes(node):
                child.parent = node

        for node in ast.walk(tree):
            parent = getattr(node, "parent", None)

            if isinstance(node, ast.ClassDef) and isinstance(parent, ast.Module):
                text = ast.get_source_segment(code, node)
                if text:
                    chunks.append(Document(
                        page_content=text,
                        metadata={"source": source, "type": "class", "name": node.name},
                    ))

            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and isinstance(parent, ast.Module):
                text = ast.get_source_segment(code, node)
                if text:
                    chunks.append(Document(
                        page_content=text,
                        metadata={"source": source, "type": "function", "name": node.name},
                    ))

    return chunks


# ---------------------------------------------------------------------------
# 3. EMBED & STORE
# ---------------------------------------------------------------------------

def build_vector_store(chunks: list) -> Chroma:
    embeddings = LangchainOpenAIEmbeddings(model="text-embedding-3-small")
    return Chroma.from_documents(chunks, embedding=embeddings)


# ---------------------------------------------------------------------------
# 4. RETRIEVER + AGENT  (identical to 3_semanticrag_ast.py)
# ---------------------------------------------------------------------------

def build_agent(vector_store: Chroma):
    retriever_tool = create_retriever_tool(
        vector_store.as_retriever(search_kwargs={"k": 1}),
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
    args = parser.parse_args()
    repo_path = str(Path(args.repo).resolve())

    docs = load_codebase(repo_path)
    chunks = chunk_code(docs)
    print(f"Loaded {len(docs)} files → {len(chunks)} chunks (AST-based)")

    vector_store = build_vector_store(chunks)
    agent = build_agent(vector_store)

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

        print(f"Q: {tc['question']}")
        print(f"\nContexts retrieved ({len(contexts)}):")
        for i, ctx in enumerate(contexts, 1):
            print(f"  [{i}] {ctx[:200]}{'...' if len(ctx) > 200 else ''}")
        print(f"\nAnswer: {answer}")

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

    print("\nRAGAS SCORECARD — AST-chunked Semantic RAG on sample_project")
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
