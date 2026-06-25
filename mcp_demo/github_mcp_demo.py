from dotenv import load_dotenv
load_dotenv()
import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents import create_agent
import os


GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
async def run_agent():
  client = MultiServerMCPClient(
      {
          "github": {
              "command": "npx",
              "args": [
                  "-y",
                  "@modelcontextprotocol/server-github"
              ],
              "env": {
                  "GITHUB_PERSONAL_ACCESS_TOKEN": GITHUB_TOKEN
              },
              "transport": "stdio"
          },
          "filesystem": {
              "command": "npx",
              "args": [
                  "-y",
                  "@modelcontextprotocol/server-filesystem",
                  "/Users/keertipurswani/GitHub/Agentic_AI_System_Design_Course/mcp_demo"
              ],
              "transport":"stdio"
          }

      }
  )
  tools = await client.get_tools()

  agent = create_agent("openai:gpt-4o", tools)
  response = await agent.ainvoke({"messages": "what are the files present in repository keertipurswani/EducosysGenerativeAI"})
  print(response["messages"][-1].content)


if __name__ == "__main__":
  asyncio.run(run_agent())
