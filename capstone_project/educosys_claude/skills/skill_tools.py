from __future__ import annotations


from pathlib import Path


from langchain.tools import tool


from educosys_claude.config import config
from educosys_claude.skills.registry import SkillRegistry, SkillNotFoundError
from educosys_claude.observability.logger import get_logger


logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Singleton registry
# ---------------------------------------------------------------------------
# One SkillRegistry instance is shared for the entire process lifetime.
# It is created lazily on the first call to get_registry() — not at import time —
# so that Path.cwd() resolves to the target project directory, not the
# educosys_claude package directory.
#
# State transitions:
#   None  ──(first call to get_registry())──►  SkillRegistry (loaded)
#                                                     │
#                              all subsequent calls return this same instance
# ---------------------------------------------------------------------------
_registry: SkillRegistry | None = None




def get_registry() -> SkillRegistry:
   """
   Return the process-wide SkillRegistry, building it on first call.


   skills_dir resolution order:
     1. Read "skills.skills_dir" from config.yaml  →  e.g. ".educosys/skills"
     2. Resolve relative to Path.cwd()             →  the target project root
        (educosys_claude is invoked from the project being analysed, so cwd()
         IS the target project, not the educosys package itself)


   Example:
     User runs educosys_claude from /home/user/my_project/
     config.yaml says skills_dir: .educosys/skills
     → skills are loaded from /home/user/my_project/.educosys/skills/
   """
   global _registry
   if _registry is None:
       skills_dir = Path.cwd() / config.get("skills", {}).get("skills_dir", ".educosys/skills")
       _registry = SkillRegistry(skills_dir)
       _registry.load()
       logger.info(f"SkillRegistry initialized from: {skills_dir}")
   return _registry




# ---------------------------------------------------------------------------
# Called by factory.py at agent build time (not a tool — plain function)
# ---------------------------------------------------------------------------


def build_skills_prompt() -> str:
   """
   Trigger registry initialisation and return the metadata-only prompt snippet.


   This is called once inside build_agent() in factory.py and its return value
   is appended to SYSTEM_PROMPT before the agent is created. The LLM therefore
   always knows what skills exist, without paying the token cost of their bodies.


   Flow:
       factory.py: build_agent()
           → build_skills_prompt()          (this function)
               → get_registry()             (init registry if first call)
                   → SkillRegistry.load()   (scan skills_dir on disk)
               → SkillRegistry.build_skills_prompt()
                   → returns compact metadata string
           → appended to SYSTEM_PROMPT
   """
   return get_registry().build_skills_prompt()




# ---------------------------------------------------------------------------
# The agent-callable tool (registered in factory.py tools list)
# ---------------------------------------------------------------------------


@tool
def load_skill(name: str) -> str:
   """
   Load the full instructions for a skill by name.
   Call this when the user's request matches one of the skills listed in your system prompt.
   Returns the skill's step-by-step instructions and lists any available support files
   (scripts, templates, resources) that you can read using the read_file tool.
   """
   # The LLM calls this tool after spotting a matching skill name in its system prompt.
   # The tool result (full SKILL.md body + support file paths) becomes part of the
   # agent's reasoning context for this query only — it is NOT permanently added to
   # the system prompt or memory.
   #
   # Three-tier disclosure for a single query:
   #   Tier 1 — system prompt:  name + description (always present, ~5 tokens/skill)
   #   Tier 2 — this tool call: full SKILL.md body (loaded on demand, ~500-2000 tokens)
   #   Tier 3 — read_file():    individual support files (only if skill instructions say to)
   logger.info(f"Tool called: load_skill('{name}')")
   try:
       return get_registry().load_skill(name)
   except SkillNotFoundError as e:
       # Return the error as a plain string so the agent can recover gracefully
       # and tell the user, rather than raising an unhandled exception.
       return str(e)
