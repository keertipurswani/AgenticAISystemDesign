from educosys_claude.config import config
from educosys_claude.observability.logger import get_logger


logger = get_logger(__name__)


def get_indexer():
   mode = config["rag"]["mode"]
   provider = config["vector_store"]["provider"]


   if mode == "hybrid" and provider == "qdrant":
       from .hybrid_qdrant import index_codebase
   elif provider == "qdrant":
       from .semantic_qdrant import index_codebase
   else:
       from .semantic_chroma import index_codebase


   return index_codebase


def get_index_inspector():
   """Return the right show_index function based on rag mode and vector_store in config."""
   mode = config["rag"]["mode"]
   provider = config["vector_store"]["provider"]


   if mode == "hybrid" and provider == "qdrant":
       from .hybrid_qdrant import show_index
   elif provider == "qdrant":
       from .semantic_qdrant import show_index
   else:
       from .semantic_chroma import show_index


   return show_index


