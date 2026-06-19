"""
Short-term memory demo — SQLite persistence + threads

- history now survives restarts (stored in SQLite)
- multiple threads: each thread_id is an independent conversation

Commands:
  /history              show messages in current thread
  /threads              list all saved thread IDs
  /switch <thread_id>   switch to a different thread
  exit                  quit
"""

import sqlite3
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from langchain_openai import ChatOpenAI

MODEL = "gpt-4o"
DB_PATH = Path("data/memory/shortterm_memory_demo.db")
llm = ChatOpenAI(model=MODEL)


# --- persistence helpers ---

def init_db(conn: sqlite3.Connection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS threads (
            thread_id TEXT NOT NULL,
            role      TEXT NOT NULL,
            content   TEXT NOT NULL
        )
    """)
    conn.commit()


def load_history(conn: sqlite3.Connection, thread_id: str) -> list[dict]:
    rows = conn.execute(
        "SELECT role, content FROM threads WHERE thread_id = ? ORDER BY rowid",
        (thread_id,),
    ).fetchall()
    history = [{"role": "system", "content": "You are a helpful assistant."}]
    history += [{"role": r, "content": c} for r, c in rows]
    return history


def save_message(conn: sqlite3.Connection, thread_id: str, role: str, content: str) -> None:
    conn.execute(
        "INSERT INTO threads (thread_id, role, content) VALUES (?, ?, ?)",
        (thread_id, role, content),
    )
    conn.commit()


def list_threads(conn: sqlite3.Connection) -> list[str]:
    rows = conn.execute(
        "SELECT DISTINCT thread_id FROM threads ORDER BY thread_id"
    ).fetchall()
    return [r[0] for r in rows]


# --- agent ---

def ask(conn: sqlite3.Connection, thread_id: str, user_input: str) -> str:
    save_message(conn, thread_id, "user", user_input)
    history = load_history(conn, thread_id)
    reply = llm.invoke(history).content
    save_message(conn, thread_id, "assistant", reply)
    return reply


def main() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    init_db(conn)

    thread_id = "default"
    print(f"Thread: {thread_id}  (use /switch <id> to change)")
    print("Commands: /history, /threads, /switch <id>, exit\n")

    while True:
        user_input = input("You: ").strip()
        if not user_input:
            continue

        if user_input.lower() == "exit":
            break

        if user_input.lower() == "/history":
            for msg in load_history(conn, thread_id):
                print(f"  [{msg['role']}] {msg['content'][:120]}")
            print()
            continue

        if user_input.lower() == "/threads":
            threads = list_threads(conn)
            print("  Saved threads:", threads or "(none yet)")
            print(f"  Active: {thread_id}\n")
            continue

        if user_input.lower().startswith("/switch "):
            new_id = user_input.split(maxsplit=1)[1].strip()
            if not new_id:
                print("Usage: /switch <thread_id>\n")
                continue
            thread_id = new_id
            print(f"Switched to thread: {thread_id}\n")
            continue

        reply = ask(conn, thread_id, user_input)
        print(f"LLM: {reply}\n")

    conn.close()


if __name__ == "__main__":
    main()
