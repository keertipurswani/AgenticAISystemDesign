from langchain.agents import create_agent

from educosys_claude.llm.factory import get_llm
from educosys_claude.agent.tools import search_codebase
from educosys_claude.observability.logger import get_logger
from educosys_claude.tools.terminal_tools import run_command, run_in_directory
from educosys_claude.tools.filesystem_tools import (
   read_file,
   write_file,
   append_file,
   list_directory,
   file_exists,
)
from educosys_claude.mcp.educosys_mcp_client import get_educosys_mcp_tools

logger = get_logger(__name__)

SYSTEM_PROMPT = """You are a senior software engineer with deep knowledge of the codebase.
Always use the search_codebase tool before answering any question.
Reference specific file names, function names and line numbers in your answers.
If you cannot find the answer in the codebase, say so explicitly."""

async def build_agent(checkpointer):
   """Create and return a LangChain agent with persistent memory."""
   llm = get_llm()
   mcp_tools = await get_educosys_mcp_tools()
   tools = [
       search_codebase,
       run_command,
       run_in_directory,
       read_file,
       write_file,
       append_file,
       list_directory,
       file_exists,
       *mcp_tools,
   ]


   return create_agent(
       llm,
       tools=tools,
       system_prompt=SYSTEM_PROMPT,
       checkpointer=checkpointer,
   )
