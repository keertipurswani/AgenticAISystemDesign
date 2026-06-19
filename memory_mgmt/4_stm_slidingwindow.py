"""
Short-term memory demo — Sliding Window


Uses @before_model middleware to trim history before every model call.
Keeps system message + last N messages. Older messages are dropped.


Tradeoff: simple and cheap, but early context is lost once the window fills.
"""

from pathlib import Path
from typing import Any
from dotenv import load_dotenv
load_dotenv()


from langchain.agents import AgentState, create_agent
from langchain.agents.middleware import before_model
from langchain.messages import RemoveMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langgraph.runtime import Runtime


MODEL = "gpt-4o"
WINDOW = 4  # number of recent messages to keep
DB_PATH = Path("data/memory/sliding_window.db")


@before_model
def sliding_window(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
   messages = state["messages"]
   if len(messages) <= WINDOW: 
       return None

   recent = messages[-WINDOW:]
   return {
       "messages": [
           RemoveMessage(id=REMOVE_ALL_MESSAGES),
           *recent,
       ]
   }


def print_thread_state(agent, thread_id: str) -> None:
   snapshot = agent.get_state({"configurable": {"thread_id": thread_id}})
   messages = (snapshot.values or {}).get("messages", [])
   print(f"\n  thread: {thread_id} | stored messages: {len(messages)}")
   for msg in messages:
       role = getattr(msg, "type", "unknown")
       content = str(getattr(msg, "content", ""))[:120]
       print(f"  [{role}] {content}")
   print()


def list_threads(checkpointer: SqliteSaver) -> list[str]:
   checkpointer.setup()
   rows = checkpointer.conn.execute(
       "SELECT DISTINCT thread_id FROM checkpoints ORDER BY thread_id"
   ).fetchall()
   return [r[0] for r in rows]



def main() -> None:
   DB_PATH.parent.mkdir(parents=True, exist_ok=True)

   with SqliteSaver.from_conn_string(str(DB_PATH)) as checkpointer:
       agent = create_agent(
           model=f"openai:{MODEL}",
           tools=[],
           system_prompt="You are a helpful assistant.",
           middleware=[sliding_window],
           checkpointer=checkpointer,
       )

       thread_id = "default"
       config = {"configurable": {"thread_id": thread_id}}


       print(f"Sliding window demo — keeping last {WINDOW} messages.")
       print("Try recalling something said early on after several turns.")
       print("Commands: /history, /threads, /switch <id>, exit\n")


       while True:
           user_input = input("You: ").strip()
           if not user_input:
               continue
           if user_input.lower() == "exit":
               break
           if user_input.lower() == "/history":
               print_thread_state(agent, thread_id)
               continue
           if user_input.lower() == "/threads":
               threads = list_threads(checkpointer)
               print(f"  Saved threads: {threads or '(none yet)'}")
               print(f"  Active: {thread_id}\n")
               continue
           if user_input.lower().startswith("/switch "):
               new_id = user_input.split(maxsplit=1)[1].strip()
               if not new_id:
                   print("Usage: /switch <thread_id>\n")
                   continue
               thread_id = new_id
               config = {"configurable": {"thread_id": thread_id}}
               print(f"Switched to thread: {thread_id}\n")
               continue


           result = agent.invoke(
               {"messages": [{"role": "user", "content": user_input}]},
               config,
           )
           print(f"LLM: {result['messages'][-1].content}\n")




if __name__ == "__main__":
   main()
