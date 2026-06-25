import os
from langchain_qdrant import QdrantVectorStore, RetrievalMode, FastEmbedSparse
from langchain_core.documents import Document
from qdrant_client import QdrantClient


from educosys_claude.config import config
from educosys_claude.context.indexers.code_parser import parse_file, get_source_files
from educosys_claude.llm.factory import get_embedder
from educosys_claude.observability.logger import get_logger


logger = get_logger(__name__)


RETRIEVAL_MODE_MAP = {
   "dense": RetrievalMode.DENSE,
   "sparse": RetrievalMode.SPARSE,
   "hybrid": RetrievalMode.HYBRID,
}



def _get_retrieval_mode() -> RetrievalMode:
   mode = config["vector_store"].get("retrieval_mode", "hybrid")
   return RETRIEVAL_MODE_MAP.get(mode, RetrievalMode.HYBRID)




def index_codebase(repo_path: str) -> QdrantVectorStore:
   """
   Parse all source files, embed with dense + sparse vectors and store in Qdrant.
   Supports dense, sparse, and hybrid retrieval modes via config.
   Skips indexing if collection already has data.
   """
   embedder = get_embedder()
   collection_name = config["qdrant"]["collection_name"]
   url = os.getenv("QDRANT_URL")
   api_key = os.getenv("QDRANT_API_KEY")
   retrieval_mode = _get_retrieval_mode()


   # Check if collection already has data
   client = QdrantClient(url=url, api_key=api_key)
   existing = [c.name for c in client.get_collections().collections]
   if collection_name in existing:
       info = client.get_collection(collection_name)
       if info.points_count > 0:
           logger.info(f"Loaded existing index with {info.points_count} chunks")
           return QdrantVectorStore.from_existing_collection(
               embedding=embedder,
               sparse_embedding=FastEmbedSparse(model_name="Qdrant/bm25"),
               retrieval_mode=retrieval_mode,
               url=url,
               api_key=api_key,
               collection_name=collection_name,
           )


   logger.info(f"Starting hybrid indexing of {repo_path}")
   files = get_source_files(repo_path)
   docs = []


   for filepath in files:
       try:
           chunks = parse_file(filepath)
       except (SyntaxError, ValueError) as e:
           logger.error(f"Skipping {filepath}: {e}")
           continue


       for chunk in chunks:
           docs.append(Document(
               page_content=chunk.content,
               metadata={
                   "source": chunk.source,
                   "name": chunk.name,
                   "type": chunk.type,
                   "start_line": chunk.start_line,
                   "end_line": chunk.end_line,
               }
           ))
           logger.debug(f"  Indexed {chunk.type} '{chunk.name}' from {filepath}")


   vector_store = QdrantVectorStore.from_documents(
       docs, embedder,
       sparse_embedding=FastEmbedSparse(model_name="Qdrant/bm25"),
       retrieval_mode=retrieval_mode,
       url=url,
       api_key=api_key,
       collection_name=collection_name,
       batch_size=50,
   )


   logger.info(f"Hybrid indexing complete. Total chunks: {len(docs)}")
   return vector_store


def show_index(vector_store: QdrantVectorStore) -> None:
   from rich.console import Console
   console = Console()
   client = vector_store.client
   collection_name = config["qdrant"]["collection_name"]
   results = client.scroll(collection_name=collection_name, with_payload=True, with_vectors=False, limit=1000)
   points = results[0]
   console.print(f"\n[bold]Hybrid Index — {len(points)} chunks[/bold]\n")


   for i, point in enumerate(points):
       content = point.payload.get("page_content", "")
       meta = point.payload.get("metadata", {})
       console.print(f"[bold cyan]── Chunk {i+1} ──────────────────────────[/bold cyan]")
       console.print(f"  File     : {meta.get('source', 'unknown')}")
       console.print(f"  Name     : {meta.get('name', 'unknown')} ({meta.get('type', 'unknown')})")
       console.print(f"  Lines    : {meta.get('start_line')} - {meta.get('end_line')}")
       console.print(f"  Code     :\n[dim]{content[:300]}[/dim]\n")
