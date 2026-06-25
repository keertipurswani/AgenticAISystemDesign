import os
from langchain_qdrant import QdrantVectorStore, RetrievalMode, FastEmbedSparse


from educosys_claude.config import config
from educosys_claude.llm.factory import get_embedder
from educosys_claude.observability.logger import get_logger


logger = get_logger(__name__)


RETRIEVAL_MODE_MAP = {
   "dense": RetrievalMode.DENSE,
   "sparse": RetrievalMode.SPARSE,
   "hybrid": RetrievalMode.HYBRID,
}


def retrieve(query: str, k: int = 5) -> list[dict]:
   """
   Retrieve top-k chunks using dense, sparse, or hybrid mode — controlled by config.
   """
   embedder = get_embedder()
   collection_name = config["qdrant"]["collection_name"]
   mode = config["vector_store"].get("retrieval_mode", "hybrid")
   retrieval_mode = RETRIEVAL_MODE_MAP.get(mode, RetrievalMode.HYBRID)


   vector_store = QdrantVectorStore.from_existing_collection(
       embedding=embedder,
       sparse_embedding=FastEmbedSparse(model_name="Qdrant/bm25"),
       retrieval_mode=retrieval_mode,
       url=os.getenv("QDRANT_URL"),
       api_key=os.getenv("QDRANT_API_KEY"),
       collection_name=collection_name,
   )


   logger.info(f"Retrieving top {k} chunks — mode: {mode} — query: {query}")
   results = vector_store.similarity_search_with_score(query, k=k)


   chunks = []
   for doc, score in results:
       meta = doc.metadata
       chunks.append({
           "content": doc.page_content,
           "source": meta["source"],
           "name": meta["name"],
           "type": meta["type"],
           "start_line": meta["start_line"],
           "end_line": meta["end_line"],
           "distance": score,
       })
       logger.debug(f"  Retrieved {meta['type']} '{meta['name']}' from {meta['source']} (score: {score:.4f})")


   logger.info(f"Retrieved {len(chunks)} chunks")
   return chunks
