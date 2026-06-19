"""
Short-term memory demo
Conversation history is appended and sent every call
The history lives in-process (a plain list). It resets on restart.
"""


from dotenv import load_dotenv
load_dotenv()


from langchain_openai import ChatOpenAI
MODEL = "gpt-4o"
llm = ChatOpenAI(model=MODEL)


# In-memory history — grows with every turn
history: list[dict] = [
   {"role": "system", "content": "You are a helpful assistant."},
]


def ask(user_input: str) -> str:
   history.append({"role": "user", "content": user_input})
   response = llm.invoke(history)
   reply = response.content
   history.append({"role": "assistant", "content": reply})
   return reply


def main() -> None:
   print("Short-term memory demo — full history is sent on every call.")
   print("Type 'exit' to quit, '/history' to see stored messages.\n")


   while True:
       user_input = input("You: ").strip()
       if user_input.lower() == "exit":
           break
       if not user_input:
           continue
       if user_input.lower() == "/history":
           for msg in history:
               print(f"  [{msg['role']}] {msg['content'][:120]}")
           print()
           continue
       reply = ask(user_input)
       print(f"LLM: {reply}\n")


if __name__ == "__main__":
   main()
