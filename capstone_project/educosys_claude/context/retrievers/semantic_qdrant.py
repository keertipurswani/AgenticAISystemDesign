import os
from langchain_qdrant import QdrantVectorStore


from educosys_claude.config import config
from educosys_claude.llm.factory import get_embedder
from educosys_claude.observability.logger import get_logger


logger = get_logger(__name__)


def retrieve(query: str, k: int = 5) -> list[dict]:
   """
   Embed the query and find the k most similar chunks in Qdrant.
   Returns a list of results with content and metadata.
   """
   embedder = get_embedder()


   vector_store = QdrantVectorStore.from_existing_collection(
       embedding=embedder,
       url=os.getenv("QDRANT_URL"),
       api_key=os.getenv("QDRANT_API_KEY"),
       collection_name=config["qdrant"]["collection_name"],
   )


   logger.info(f"Retrieving top {k} chunks for query: {query}")
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
