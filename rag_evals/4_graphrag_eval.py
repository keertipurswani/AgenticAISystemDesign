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

import networkx as nx
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings as LangchainOpenAIEmbeddings
from pydantic import BaseModel, Field

from ragas import EvaluationDataset, SingleTurnSample, evaluate
from ragas.metrics._context_precision import LLMContextPrecisionWithReference
from ragas.metrics._context_recall import LLMContextRecall
from ragas.metrics._faithfulness import Faithfulness
from ragas.metrics._answer_relevance import AnswerRelevancy
from ragas.metrics._factual_correctness import FactualCorrectness
from ragas.llms import llm_factory


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CODEBASE = PROJECT_ROOT / "sample_project"

CODE_EXTENSIONS = {".py", ".ts", ".js", ".java", ".go", ".rs", ".md"}
SKIP_DIRS = {".git", ".venv", "venv", "__pycache__", "node_modules", ".mypy_cache", ".ruff_cache"}


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class CodeRelationship(BaseModel):
    subject: str = Field(description="Class, function, module, or file path")
    predicate: str = Field(
        description="Relationship type (e.g., DEFINES, IMPORTS, USES, CALLS, DEPENDS_ON, INHERITS_FROM)"
    )
    obj: str = Field(description="Target class, function, module, or file path")


class GraphDocument(BaseModel):
    relationships: list[CodeRelationship] = Field(description="All code relationships extracted from the source file")


class Entities(BaseModel):
    names: list[str] = Field(
        description="Code entities in the query: class names, function names, modules, file paths"
    )


# ---------------------------------------------------------------------------
# 1. LOAD
# ---------------------------------------------------------------------------

def load_codebase(root: Path) -> list[tuple[str, str]]:
    root = root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Codebase path does not exist: {root}")
    files = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in CODE_EXTENSIONS:
            continue
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        files.append((str(path.relative_to(root)), path.read_text(encoding="utf-8")))
    return files


# ---------------------------------------------------------------------------
# 2. EXTRACT RELATIONSHIPS & BUILD GRAPH
# ---------------------------------------------------------------------------

def extract_relationships(relationship_extractor, files: list[tuple[str, str]]) -> list[CodeRelationship]:
    relationships = []
    for rel_path, content in files:
        result = relationship_extractor.invoke(
            {"messages": [{"role": "user", "content": f"File: {rel_path}\n\n{content}"}]}
        )
        relationships.extend(result["structured_response"].relationships)
    return relationships


def build_graph(relationships: list[CodeRelationship]) -> nx.DiGraph:
    graph = nx.DiGraph()
    for r in relationships:
        graph.add_edge(r.subject.strip(), r.obj.strip(), relation=r.predicate)
    return graph


# ---------------------------------------------------------------------------
# 3. RETRIEVE
# ---------------------------------------------------------------------------

def match_nodes(graph: nx.DiGraph, entity: str) -> list[str]:
    needle = entity.strip().lower()
    return [node for node in graph.nodes() if needle in node.lower() or node.lower() in needle]


def graph_retrieve(graph: nx.DiGraph, entity_extractor, query: str, depth: int = 2) -> str:
    result = entity_extractor.invoke({"messages": [{"role": "user", "content": query}]})
    entities = result["structured_response"].names

    relationships = []
    for entity in entities:
        for node in match_nodes(graph, entity):
            neighbourhood = nx.ego_graph(graph, node, radius=depth, undirected=True)
            for source, target, data in neighbourhood.edges(data=True):
                relationships.append(f"{source} -[{data['relation']}]-> {target}")

    if not relationships:
        return "No relevant graph data found."
    return "Knowledge Graph context:\n" + "\n".join(sorted(set(relationships)))


# ---------------------------------------------------------------------------
# 4. RUN
# ---------------------------------------------------------------------------

def run_query(qa_agent, entity_extractor, graph: nx.DiGraph, question: str, depth: int = 2) -> tuple[str, list[str]]:
    context = graph_retrieve(graph, entity_extractor, question, depth=depth)
    result = qa_agent.invoke(
        {"messages": [{"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}]}
    )
    answer = result["messages"][-1].content
    return answer, [context]


# ---------------------------------------------------------------------------
# Golden Dataset
# ---------------------------------------------------------------------------

from golden_dataset import TEST_CASES


# ---------------------------------------------------------------------------
# 5. MAIN
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=DEFAULT_CODEBASE)
    parser.add_argument("--depth", type=int, default=2, help="Graph neighbourhood radius (default: 2)")
    args = parser.parse_args()

    files = load_codebase(args.repo.resolve())
    if not files:
        raise SystemExit(f"No source files found under {args.repo}")

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    relationship_extractor = create_agent(
        model=llm,
        tools=[],
        response_format=GraphDocument,
        system_prompt=(
            "Extract code relationships from the source file as (subject, predicate, object) facts.\n"
            "Use ALL_CAPS predicates such as: DEFINES, IMPORTS, USES, CALLS, DEPENDS_ON, "
            "INHERITS_FROM, IMPLEMENTS, VALIDATES, SENDS_TO, CONFIGURES.\n"
            "Subjects and objects should be class names, function names, module paths, or file paths.\n"
            "Capture imports, constructor dependencies, method calls, and cross-module relationships.\n"
            "Be consistent: use the same name for the same class or module across relationships."
        ),
    )
    entity_extractor = create_agent(
        model=llm,
        tools=[],
        response_format=Entities,
        system_prompt="Extract code-related entities from the user message: class names, function names, module names, and file paths.",
    )
    qa_agent = create_agent(
        model=llm,
        tools=[],
        system_prompt=(
            "You are a codebase assistant. Answer ONLY from the context provided in the user message. "
            "Reference specific classes, files, and relationships when possible. "
            "If you cannot answer from context, say so."
        ),
    )

    print(f"Loaded {len(files)} files — extracting knowledge graph...")
    relationships = extract_relationships(relationship_extractor, files)
    graph = build_graph(relationships)
    print(f"Graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")

    # Collect GraphRAG outputs
    print("\nRunning agent on test cases...")
    samples = []
    for tc in TEST_CASES:
        answer, contexts = run_query(qa_agent, entity_extractor, graph, tc["question"], depth=args.depth)
        samples.append(SingleTurnSample(
            user_input=tc["question"],
            response=answer,
            retrieved_contexts=contexts,
            reference=tc["reference"],
        ))

        print(f"\nQ: {tc['question']}")
        print(f"Context retrieved:")
        print(f"  {contexts[0][:200]}{'...' if len(contexts[0]) > 200 else ''}")
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

    print("\nRAGAS SCORECARD — GraphRAG on sample_project")
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
