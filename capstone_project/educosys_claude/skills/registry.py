from __future__ import annotations
import re
from pathlib import Path

from educosys_claude.observability.logger import get_logger
logger = get_logger(__name__)

# Every skill package must have this file at its root.
# Folders inside skills_dir that lack this file are skipped entirely.
SKILL_FILENAME = "SKILL.md"

class SkillNotFoundError(Exception):
   pass

class SkillRegistry:
   """
   Owns the in-memory catalog of all skills found in skills_dir.

   Expected layout on disk:
       skills_dir/
           python_debug/
               SKILL.md          ← required — frontmatter + instructions
               scripts/          ← optional — executable helpers
               templates/        ← optional — output templates
               resources/        ← optional — reference docs
           write_tests/
               SKILL.md
               templates/
                   pytest_template.py


   After calling load(), the internal _skills dict looks like:
       {
           "python_debug": {
               "meta":      {"name": "python_debug", "description": "...", ...},
               "body":      "You are an expert Python debugger. Follow these steps...",
               "skill_dir": Path(".educosys/skills/python_debug")
           },
           "write_tests": {
               "meta":      {"name": "write_tests", "description": "...", ...},
               "body":      "Generate pytest unit tests...",
               "skill_dir": Path(".educosys/skills/write_tests")
           }
       }

   "meta" is parsed from the YAML frontmatter block (name, description, when_to_use, ...).
   "body" is everything after the frontmatter — the actual instructions sent to the LLM.
   "skill_dir" is kept so we can later discover support files inside that folder.
   """

   def __init__(self, skills_dir: Path) -> None:
       self._skills_dir = skills_dir
       self._skills: dict[str, dict] = {}  # populated by load()

   # ------------------------------------------------------------------
   # Startup: scan disk and populate _skills
   # ------------------------------------------------------------------

   def load(self) -> None:
       """
       Walk skills_dir and parse every SKILL.md found.
       Called once at agent startup via get_registry() in skill_tools.py.
       Safe to call again to hot-reload if skills change on disk.
       """
       self._skills.clear()
       if not self._skills_dir.exists():
           logger.warning(f"skills_dir not found: {self._skills_dir}. No skills loaded.")
           return

       for skill_dir in self._skills_dir.iterdir():
           if not skill_dir.is_dir():
               continue  # skip stray files at the top level of skills_dir

           skill_file = skill_dir / SKILL_FILENAME
           if not skill_file.exists():
               logger.warning(f"Skipping {skill_dir.name}: no SKILL.md found")
               continue

           try:
               meta, body = _parse_skill_file(skill_file)
               # Use "name" from frontmatter if declared; fall back to the folder name.
               # This means the folder name and the frontmatter name should match,
               # but the frontmatter value is what the agent will refer to.
               name = meta.get("name", skill_dir.name)
               self._skills[name] = {
                   "meta": meta,       # dict of frontmatter fields
                   "body": body,       # raw instruction text (sent to LLM on activation)
                   "skill_dir": skill_dir,  # kept for support file discovery
               }
               logger.info(f"Loaded skill: {name}")
           except Exception as e:
               logger.error(f"Failed to load skill from {skill_dir.name}: {e}")

   # ------------------------------------------------------------------
   # Called at agent build time → injected into SYSTEM_PROMPT
   # ------------------------------------------------------------------

   def build_skills_prompt(self) -> str:
       """
       Returns a compact string appended to the agent's SYSTEM_PROMPT at startup.


       ONLY includes metadata (name, description, when_to_use) — NOT the full
       skill body. This keeps the system prompt small. The agent loads full
       instructions on demand by calling the load_skill tool (progressive disclosure).


       Example output injected into SYSTEM_PROMPT:
           === Available Skills ===
           - python_debug: Debug Python errors and tracebacks | when_to_use: error, traceback, exception
           - write_tests: Generate pytest unit tests | when_to_use: test, coverage, pytest


           When the user's request matches a skill, call load_skill(name) ...
       """
       if not self._skills:
           return ""

       lines = ["=== Available Skills ==="]
       for name, skill in self._skills.items():
           meta = skill["meta"]
           description = meta.get("description", "No description provided.")
           when_to_use = meta.get("when_to_use", "")
           line = f"- {name}: {description}"
           if when_to_use:
               line += f" | when_to_use: {when_to_use}"
           lines.append(line)

       # This sentence is the agent's instruction on how to act when it recognises a skill.
       lines.append(
           "\nWhen the user's request matches a skill, call load_skill(name) "
           "to get the full instructions before proceeding."
       )
       return "\n".join(lines)

   # ------------------------------------------------------------------
   # Called at query time → triggered by the agent via the load_skill tool
   # ------------------------------------------------------------------


   def load_skill(self, name: str) -> str:
       """
       Returns the full SKILL.md body for the named skill, plus a listing of
       any support files inside the skill's folder.

       The agent calls this (via the load_skill @tool in skill_tools.py) after
       spotting a matching skill in the system prompt. The returned text becomes
       part of the agent's reasoning context for that query.

       Support files are listed but NOT read here — the agent decides which ones
       it needs and calls read_file() on them individually (tier-3 disclosure).

       Example return value:
           You are an expert Python debugger. Follow these steps...
           1. Ask for the full traceback if not provided.
           ...

           --- Support Files Available ---
             /path/to/skills/python_debug/scripts/run_debugger.py
             /path/to/skills/python_debug/resources/common_errors.md
           You can read any of these files using the read_file tool ...
       """
       if name not in self._skills:
           available = ", ".join(self._skills.keys()) or "none"
           raise SkillNotFoundError(
               f"Skill '{name}' not found. Available skills: {available}"
           )

       skill = self._skills[name]
       body = skill["body"]
       skill_dir: Path = skill["skill_dir"]

       # Discover support files and append their paths to the response.
       # The agent can then call read_file() on whichever ones are relevant.
       support_files = _list_support_files(skill_dir)
       result = body
       if support_files:
           result += "\n\n--- Support Files Available ---\n"
           result += "\n".join(f"  {p}" for p in support_files)
           result += (
               "\nYou can read any of these files using the read_file tool "
               "if the skill instructions reference them."
           )
       return result

   @property
   def skill_names(self) -> list[str]:
       return list(self._skills.keys())


# ---------------------------------------------------------------------------
# Module-level helpers (private to this file)
# ---------------------------------------------------------------------------


def _parse_skill_file(path: Path) -> tuple[dict, str]:
   """
   Read a SKILL.md and split it into (frontmatter_dict, body_str).
   A valid SKILL.md looks like:
       ---
       name: python_debug
       description: Debug Python errors and tracebacks
       when_to_use: error, traceback, exception
       ---
       You are an expert Python debugger...   ← this is the body

   The regex matches the opening ---, captures everything up to the closing ---,
   then captures the rest of the file as the body.
   If there is no frontmatter, the entire file is treated as the body and
   meta is returned as an empty dict (the folder name becomes the skill name).
   """
   text = path.read_text(encoding="utf-8")

   match = re.match(r"^---\s*\n(.*?)\n---\s*\n(.*)", text, re.DOTALL)
   if not match:
       # No frontmatter block found — body is the whole file.
       return {}, text.strip()


   frontmatter_str = match.group(1)  # raw YAML text between the --- markers
   body = match.group(2).strip()     # everything after the closing ---
   meta = _parse_yaml(frontmatter_str)
   return meta, body




def _parse_yaml(text: str) -> dict:
   """
   Parse the frontmatter YAML string into a dict.


   Strategy:
     1. Try yaml.safe_load (pyyaml is already a dependency via config.py).
     2. If yaml isn't importable for any reason, fall back to a line-by-line
        parser that handles flat "key: value" pairs only (no nesting or lists).
        This is enough for the required frontmatter fields.
   """
   try:
       import yaml  # already present — loaded by config.py at startup
       return yaml.safe_load(text) or {}
   except ImportError:
       pass


   # Minimal fallback for flat key: value frontmatter
   result = {}
   for line in text.splitlines():
       if ":" in line:
           key, _, value = line.partition(":")
           result[key.strip()] = value.strip()
   return result


def _list_support_files(skill_dir: Path) -> list[str]:
   """
   Return the absolute paths of every file inside the skill folder except SKILL.md.

   rglob("*") descends into all subdirectories (scripts/, templates/, resources/, ...),
   so the agent gets a complete picture of what resources the skill package ships with.

   Example for python_debug/:
       [
           "/path/to/skills/python_debug/scripts/run_debugger.py",
           "/path/to/skills/python_debug/resources/common_errors.md"
       ]
   """
   files = []
   for f in skill_dir.rglob("*"):
       if f.is_file() and f.name != SKILL_FILENAME:
           files.append(str(f))
   return files
