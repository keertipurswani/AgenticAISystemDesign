import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain.agents.middleware import SummarizationMiddleware
from pathlib import Path


from educosys_claude.config import config
from educosys_claude.llm.factory import get_llm
from educosys_claude.observability.logger import get_logger


logger = get_logger(__name__)




def get_checkpointer() -> SqliteSaver:
   db_path = config["memory"]["db_path"]
   Path(db_path).parent.mkdir(exist_ok=True)
   logger.info(f"Using SQLite checkpointer at {db_path}")
   conn = sqlite3.connect(db_path, check_same_thread=False)
   return SqliteSaver(conn)




def get_session_history(thread_id: str) -> list[dict]:
   checkpointer = get_checkpointer()
   config_ = {"configurable": {"thread_id": thread_id}}
   checkpoint = checkpointer.get(config_)
   if not checkpoint:
       return []
   messages = checkpoint["channel_values"].get("messages", [])
   return [
       {"role": "user" if m.type == "human" else "assistant", "content": m.content}
       for m in messages
   ]




def get_summarization_middleware() -> SummarizationMiddleware:
   return SummarizationMiddleware(
       model=get_llm(),
       trigger=("tokens", config["memory"]["summarize_at_tokens"]),
       keep=("messages", config["memory"]["keep_last_messages"]),
   )


