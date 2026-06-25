import chromadb


from educosys_claude.config import config
from educosys_claude.context.indexers.code_parser import parse_file, get_source_files
from educosys_claude.llm.factory import get_embedder
from educosys_claude.observability.logger import get_logger


logger = get_logger(__name__)


def index_codebase(repo_path: str) -> chromadb.Collection:
   """
   Parse all source files in repo_path, embed each chunk and store in ChromaDB.
   Returns the ChromaDB collection. Skips indexing if collection already has data.
   """
   embedder = get_embedder()
   chroma_client = chromadb.PersistentClient(path=config["chromadb"]["persist_dir"])
   collection = chroma_client.get_or_create_collection(name=config["chromadb"]["collection_name"])


   if collection.count() > 0:
       logger.info(f"Loaded existing index with {collection.count()} chunks")
       return collection


   logger.info(f"Starting semantic indexing of {repo_path}")
   files = get_source_files(repo_path)


   for filepath in files:
       try:
           chunks = parse_file(filepath)
       except (SyntaxError, ValueError) as e:
           logger.error(f"Skipping {filepath}: {e}")
           continue


       for chunk in chunks:
           embedding = embedder.embed_query(chunk.content)
           doc_id = f"{chunk.source}::{chunk.name}::{chunk.start_line}"
           collection.upsert(
               ids=[doc_id],
               embeddings=[embedding],
               documents=[chunk.content],
               metadatas=[{
                   "source": chunk.source,
                   "name": chunk.name,
                   "type": chunk.type,
                   "start_line": chunk.start_line,
                   "end_line": chunk.end_line,
               }]
           )
           logger.debug(f"  Indexed {chunk.type} '{chunk.name}' from {filepath}")


   logger.info(f"Semantic indexing complete. Total chunks: {collection.count()}")
   return collection




def show_index(collection: chromadb.Collection) -> None:
   """Display all documents stored in the ChromaDB collection."""
   from rich.console import Console
   console = Console()
   results = collection.get(include=["documents", "metadatas", "embeddings"])
   console.print(f"\n[bold]Semantic Index — {collection.count()} chunks[/bold]\n")


   for i, (doc, meta, emb) in enumerate(zip(
       results["documents"],
       results["metadatas"],
       results["embeddings"]
   )):
       console.print(f"[bold cyan]── Chunk {i+1} ──────────────────────────[/bold cyan]")
       console.print(f"  File     : {meta['source']}")
       console.print(f"  Name     : {meta['name']} ({meta['type']})")
       console.print(f"  Lines    : {meta['start_line']} - {meta['end_line']}")
       console.print(f"  Embedding: [{', '.join(f'{v:.4f}' for v in emb[:5])}...] ({len(emb)} dims)")
       console.print(f"  Code     :\n[dim]{doc[:300]}[/dim]\n")
