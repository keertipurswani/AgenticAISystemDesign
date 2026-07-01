from langchain_mcp_adapters.client import MultiServerMCPClient
from educosys_claude.mcp.educosys_mcp_config import load_educosys_mcp_configs
from educosys_claude.observability.logger import get_logger


logger = get_logger(__name__)


async def get_educosys_mcp_tools() -> list:
  """Connect to all configured MCP servers and return their tools."""
  configs = load_educosys_mcp_configs()
  logger.info(f"Connecting to MCP servers: {list(configs.keys())}")
  client = MultiServerMCPClient(configs)
  tools = await client.get_tools()
  logger.info(f"Loaded {len(tools)} tools from MCP servers")
  return tools
