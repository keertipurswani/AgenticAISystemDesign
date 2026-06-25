from educosys_claude.agent.factory import build_agent
from educosys_claude.observability.logger import get_logger


logger = get_logger(__name__)


def handle_query(question: str, thread_id: str) -> str:
   """Entry point for all user queries - builds the agent and runs it."""
   logger.info(f"Handling query for session {thread_id}: {question}")
   agent = build_agent()
   agent_config = {"configurable": {"thread_id": thread_id}}
   response = agent.invoke({"messages": [{"role": "user", "content": question}]}, agent_config)
   return response["messages"][-1].content
