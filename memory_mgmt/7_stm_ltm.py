"""
Two memory layers, two databases:
  thread_id  →  SqliteSaver  →  all messages in this chat thread (STM)
  user_id    →  SqliteStore  →  behavioral rules shared across threads (LTM)

Procedural memory = "how to behave" rules.
When the user says "always respond in bullet points" or "be very concise",
the agent detects this as a procedural rule, saves it to the store, and
applies it in every future session — including after a full restart.

Demo:
  1. Say "always respond in bullet points".
  2. Observe the agent adopts the behavior immediately.
  3. Type 'exit', rerun — the rule is still active.
  4. Use '/show-ltm' and '/show-stm' to inspect both memory layers.
  5. Run with --thread-id chat2 to start a new thread (STM resets, LTM carries over).
"""

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import ToolRuntime, tool
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.store.sqlite import SqliteStore

load_dotenv()

MODEL = "gpt-4o"

DB_DIR = Path(__file__).parent / "data" / "memory"
DB_DIR.mkdir(parents=True, exist_ok=True)

LTM_DB = str(DB_DIR / "ltm.db")
STM_DB = str(DB_DIR / "stm.db")


# --------------------------------------------------------------------------
# Context — passed on every agent.invoke(..., context=Context(...))
# create_agent injects this into ToolRuntime so tools know which user to use.
# --------------------------------------------------------------------------
@dataclass
class Context:
    user_id: str


def rules_ns(user_id: str) -> tuple:
    return (user_id, "rules")


# --------------------------------------------------------------------------
# Tools — ToolRuntime[Context] gives each tool access to:
#   runtime.store   → the SqliteStore (LTM)
#   runtime.context → the Context dataclass (user_id)
# --------------------------------------------------------------------------
@tool
def save_rule(rule: str, runtime: ToolRuntime[Context]) -> str:
    """Save a behavioral rule learned from the user.

    Call this whenever the user instructs you HOW to behave —
    tone, format, language, verbosity, etc.

    Args:
        rule: the behavioral instruction as a clear, imperative sentence
              e.g. 'Always respond in bullet points.'
    """
    user_id = runtime.context.user_id
    key = re.sub(r"[^a-z0-9]+", "-", rule.lower())[:48].strip("-") or "rule"
    runtime.store.put(rules_ns(user_id), key, {"rule": rule})
    return f"Saved procedural rule: {rule}"


@tool
def list_rules(runtime: ToolRuntime[Context]) -> str:
    """List all behavioral rules learned from the user."""
    user_id = runtime.context.user_id
    items = runtime.store.search(rules_ns(user_id), limit=50)
    if not items:
        return "No behavioral rules stored yet."
    return "\n".join(f"- {i.value['rule']}" for i in items)


TOOLS = [save_rule, list_rules]


# --------------------------------------------------------------------------
# Inspect helpers
# --------------------------------------------------------------------------
def list_threads(checkpointer: SqliteSaver) -> list[str]:
    checkpointer.setup()
    rows = checkpointer.conn.execute(
        "SELECT DISTINCT thread_id FROM checkpoints ORDER BY thread_id"
    ).fetchall()
    return [r[0] for r in rows]



def print_ltm(store: SqliteStore, user_id: str) -> None:
    store.setup()
    items = store.search(rules_ns(user_id), limit=50)
    print(f"\n=== LONG-TERM MEMORY (user_id={user_id}) ===")
    if not items:
        print("  (no rules stored yet)")
    for i in items:
        print(f"  • {i.value['rule']}")
    print()


def print_stm(agent, thread_id: str) -> None:
    snapshot = agent.get_state({"configurable": {"thread_id": thread_id}})
    messages = (snapshot.values or {}).get("messages", [])
    print(f"\n=== SHORT-TERM MEMORY (thread_id={thread_id}) ===")
    print(f"  {len(messages)} message(s) stored")
    for i, m in enumerate(messages, 1):
        role = getattr(m, "type", getattr(m, "role", "?"))
        text = str(getattr(m, "content", ""))[:120]
        print(f"  {i:02d}. [{role}] {text}")
    print()


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="create_agent + STM + LTM")
    parser.add_argument("--user-id", default="demo-user")
    parser.add_argument("--thread-id", default="thread-1")
    args = parser.parse_args()

    user_id = args.user_id
    thread_id = args.thread_id

    # thread_id routes STM (which conversation thread to restore)
    config = {"configurable": {"thread_id": thread_id}}
    # user_id routes LTM (which user's rules to load/save)
    context = Context(user_id=user_id)

    with SqliteStore.from_conn_string(LTM_DB) as store:
        store.setup()

        with SqliteSaver.from_conn_string(STM_DB) as checkpointer:

            # Load existing rules so the system prompt reflects them on startup
            existing_rules = store.search(rules_ns(user_id), limit=50)
            rules_text = (
                "\n".join(f"- {i.value['rule']}" for i in existing_rules)
                if existing_rules
                else "None yet."
            )

            system_prompt = f"""You are a helpful assistant that learns behavioral preferences.

When the user tells you HOW to behave (format, tone, language, verbosity, style),
call save_rule to persist it as a long-term procedural memory.
At the start of each response, silently apply all active rules.
If asked what rules you follow, call list_rules.

Active rules from long-term memory:
{rules_text}"""

            # create_agent handles the tool loop automatically.
            # store=      wires LTM into tools via ToolRuntime[Context].
            # checkpointer= wires STM (full message history per thread_id).
            # context_schema= tells create_agent the shape of our Context dataclass.
            agent = create_agent(
                model=f"openai:{MODEL}",
                tools=TOOLS,
                system_prompt=system_prompt,
                store=store,
                checkpointer=checkpointer,
                context_schema=Context,
            )

            print("create_agent + SqliteSaver (STM) + SqliteStore (LTM).")
            print(f"LTM DB : {LTM_DB}")
            print(f"STM DB : {STM_DB}")
            print(f"user_id={user_id}  |  thread_id={thread_id}")
            if existing_rules:
                print(f"Loaded {len(existing_rules)} rule(s) from previous sessions.")
            print("Commands: '/show-ltm'  '/show-stm'  '/threads'  '/switch <thread-id>'  'exit'\n")

            while True:
                user_input = input("You: ").strip()
                if user_input.lower() == "exit":
                    break
                if not user_input:
                    continue
                if user_input.lower() == "/show-ltm":
                    print_ltm(store, user_id)
                    continue
                if user_input.lower() == "/show-stm":
                    print_stm(agent, thread_id)
                    continue
                if user_input.lower() == "/threads":
                    threads = list_threads(checkpointer)
                    print(f"  Saved threads: {threads or '(none yet)'}")
                    print(f"  Active: {thread_id}\n")
                    continue
                if user_input.lower().startswith("/switch "):
                    new_thread = user_input.split(maxsplit=1)[1].strip()
                    if not new_thread:
                        print("Usage: /switch <thread-id>\n")
                        continue
                    thread_id = new_thread
                    config = {"configurable": {"thread_id": thread_id}}
                    print(f"Switched to thread: {thread_id}  (LTM rules unchanged)\n")
                    continue

                result = agent.invoke(
                    {"messages": [{"role": "user", "content": user_input}]},
                    config=config,
                    context=context,
                )
                reply = result["messages"][-1].content
                print(f"Agent: {reply}\n")


if __name__ == "__main__":
    main()
