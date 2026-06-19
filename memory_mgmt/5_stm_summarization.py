"""
Short-term memory demo — Summarization

Custom @before_model middleware that compresses older messages once the
message count threshold is reached. Recent messages are kept raw alongside
the summary. Prints when summarization is triggered.

Tradeoff: retains gist of early context, but fine details may be lost.

Commands:
  /history              show stored thread state
  /threads              list all saved thread IDs
  /switch <thread_id>   switch to a different thread
  exit                  quit
"""

from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()

from langchain.agents import AgentState, create_agent
from langchain.agents.middleware import before_model
from langchain.messages import RemoveMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langgraph.runtime import Runtime

MODEL = "gpt-4o"
SUMMARIZE_AFTER = 4  # summarize when stored messages exceed this count
KEEP_RECENT = 1      # keep last N messages raw after summarization
DB_PATH = Path("data/memory/summarization.db")

summarizer = ChatOpenAI(model=MODEL)


@before_model
def summarize_middleware(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    messages = state["messages"]
    if len(messages) <= SUMMARIZE_AFTER:
        return None

    older = messages[:-KEEP_RECENT]
    recent = messages[-KEEP_RECENT:]

    conversation_text = "\n".join(
        f"{getattr(m, 'type', 'unknown')}: {getattr(m, 'content', '')}"
        for m in older
    )
    summary = summarizer.invoke(
        f"Summarize this conversation in 2-3 sentences, preserving key facts:\n\n{conversation_text}"
    ).content

    print(f"\n  [summarization triggered — compressed {len(older)} messages into summary]")
    print(f"  [summary: {summary[:120]}...]\n")

    return {
        "messages": [
            RemoveMessage(id=REMOVE_ALL_MESSAGES),
            {"role": "system", "content": f"Conversation summary so far:\n{summary}"},
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
            middleware=[summarize_middleware],
            checkpointer=checkpointer,
        )

        thread_id = "default"
        config = {"configurable": {"thread_id": thread_id}}

        print(f"Summarization demo — summarizes after {SUMMARIZE_AFTER} messages, keeps last {KEEP_RECENT} raw.")
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
