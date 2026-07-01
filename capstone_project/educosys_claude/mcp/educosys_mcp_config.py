import os
import re
import json
from pathlib import Path
from dotenv import load_dotenv


load_dotenv()


_CONFIG_PATH = Path(__file__).parent.parent / "educosys_mcp_servers.json"


def load_educosys_mcp_configs() -> dict:
 """Return mcp_servers dict from educosys_mcp_servers.json with env vars resolved.


 CWD is injected at load time so ${CWD} in the config always resolves to the
 directory where educosys claude was launched — not the package install location.
 """
 os.environ.setdefault("CWD", str(Path.cwd()))
 raw = json.loads(_CONFIG_PATH.read_text())
 # Replace ${VAR} placeholders in the config with actual env var values
 resolved = re.sub(r"\$\{(\w+)\}", lambda m: os.getenv(m.group(1), ""), json.dumps(raw))
 return json.loads(resolved).get("mcp_servers", {})
