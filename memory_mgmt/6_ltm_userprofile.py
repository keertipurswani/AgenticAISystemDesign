"""
Long-term memory 
Introduces SqliteStore, LangGraph's durable key-value store for LTM.
No tools, no agent — just raw store.put / store.get / store.search.


Two primitives to learn:
 namespace  = tuple path (like a folder)  e.g. ("users",)
 key        = string within the namespace  e.g. "alice"
 value      = any JSON-serialisable dict


Architecture this file teaches:
   user input
       └─► LLM answers
       └─► second LLM call extracts facts
       └─► store.put(namespace, key, facts)   # persisted to SQLite


On restart:
   store.get(namespace, key)                  # load facts back
       └─► inject into system prompt
       └─► LLM answers with context
Demo:
 1. Tell the agent your name, job, and a preference.
 2. Type 'exit'.
 3. Rerun — the agent still knows you.
 4. Use '/facts' to inspect what is stored.
"""


import json
from pathlib import Path


from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.store.sqlite import SqliteStore


load_dotenv()
MODEL = "gpt-4o"
llm = ChatOpenAI(model=MODEL)


# --------------------------------------------------------------------------
# Storage layout
#   namespace ("users",)  key <user_id>  →  profile dict
#   namespace (<user_id>, "facts")  key <slug>  →  individual fact
# --------------------------------------------------------------------------
USERS_NS = ("users",)
DB_PATH = Path(__file__).parent / "data" / "memory" / "ltm_store.db"
DB_PATH.parent.mkdir(parents=True, exist_ok=True)


USER_ID = "demo-user"


def facts_namespace(user_id: str) -> tuple:
   return (user_id, "facts")


# --------------------------------------------------------------------------
# Fact extraction — a plain second LLM call, not a tool
# --------------------------------------------------------------------------
def extract_facts(user_input: str, reply: str, existing: dict) -> dict:
   prompt = f"""You are a fact extractor.
Extract any durable facts about the user from this conversation turn
(name, job, preferences, location, constraints, goals — anything worth
remembering long-term).


Existing facts already stored:
{json.dumps(existing, indent=2)}


User said : {user_input}
Assistant : {reply}


Return a single JSON object that MERGES existing facts with any NEW facts
you found. If there is nothing new, return existing facts unchanged.
Return ONLY valid JSON — no markdown, no explanation."""


   result = llm.invoke([{"role": "user", "content": prompt}])
   try:
       return json.loads(result.content)
   except json.JSONDecodeError:
       return existing




def build_system_prompt(facts: dict) -> str:
   base = "You are a helpful assistant with long-term memory."
   if not facts:
       return base
   lines = "\n".join(f"  - {k}: {v}" for k, v in facts.items())
   return f"{base}\n\nWhat you know about the user:\n{lines}"




# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main() -> None:
   # SqliteStore is a context manager — keeps the DB connection alive.
   with SqliteStore.from_conn_string(str(DB_PATH)) as store:
       store.setup()


       # Load existing facts for this user from LTM
       item = store.get(USERS_NS, USER_ID)
       facts: dict = dict(item.value) if item else {}


       history = [{"role": "system", "content": build_system_prompt(facts)}]


       print("Step 1 — raw SqliteStore (no tools, no agent).")
       print(f"DB: {DB_PATH}")
       print(f"user_id: {USER_ID}")
       if facts:
           print(f"Loaded {len(facts)} fact(s) from previous session.")
       print("Commands: '/facts'  'exit'\n")


       while True:
           user_input = input("You: ").strip()
           if user_input.lower() == "exit":
               break
           if not user_input:
               continue
           if user_input.lower() == "/facts":
               # store.get returns an Item; .value is the stored dict
               item = store.get(USERS_NS, USER_ID)
               print(json.dumps(dict(item.value) if item else {}, indent=2))
               print()
               continue


           history.append({"role": "user", "content": user_input})
           response = llm.invoke(history)
           reply = response.content
           history.append({"role": "assistant", "content": reply})
           print(f"LLM: {reply}\n")


           # Extract new facts and persist — this is what a tool will do in step 2
           updated = extract_facts(user_input, reply, facts)
           if updated != facts:
               facts = updated
               # store.put(namespace, key, value)
               # Calling put again on same key is an upsert — it overwrites.
               store.put(USERS_NS, USER_ID, facts)
               print(f"  [stored {len(facts)} fact(s) to SqliteStore]\n")




if __name__ == "__main__":
   main()


