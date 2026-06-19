"""
Stateless LLM demo
Shows that a raw LLM call has NO memory between invocations.
Each call is a fresh context — previous messages are discarded.
"""

from dotenv import load_dotenv
load_dotenv()


from langchain_openai import ChatOpenAI

MODEL = "gpt-4o"
llm = ChatOpenAI(model=MODEL)


def ask(messages: list[dict]) -> str:
   response = llm.invoke(messages)
   return response.content


def main() -> None:
   print("Type 'exit' to quit.\n")


   while True:
       user_input = input("You: ").strip()
       if user_input.lower() == "exit":
           break
       if not user_input:
           continue


       # Every call gets only the current message — no conversation history
       messages = [
           {"role": "system", "content": "You are a helpful assistant."},
           {"role": "user", "content": user_input},
       ]
       reply = ask(messages)
       print(f"LLM: {reply}\n")


if __name__ == "__main__":
   main()
