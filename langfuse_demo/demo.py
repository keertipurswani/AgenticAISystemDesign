from dotenv import load_dotenv
load_dotenv()
from langchain.agents import create_agent

from langfuse import get_client
from langfuse.langchain import CallbackHandler

# Initialize Langfuse client
langfuse = get_client()
# Initialize Langfuse CallbackHandler for Langchain (tracing)
langfuse_handler = CallbackHandler()


def run_agent():
  agent = create_agent("openai:gpt-4o")
  response = agent.invoke({"messages": "who is dhoni?"}, config={"callbacks": [langfuse_handler]})
  print(response["messages"][-1].content)


if __name__ == "__main__":
  run_agent()
