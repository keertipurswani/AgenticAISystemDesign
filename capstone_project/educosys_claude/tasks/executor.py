from __future__ import annotations

import json
from pydantic import BaseModel

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model

from educosys_claude.config import config
from educosys_claude.tools.filesystem_tools import (
    append_file,
    file_exists,
    list_directory,
    read_file,
    write_file,
)
from educosys_claude.tools.terminal_tools import run_command
from educosys_claude.observability.logger import get_logger

logger = get_logger(__name__)


# Each task type gets a minimal, focused toolset — least privilege per task.
_TOOLS_BY_TYPE: dict[str, list] = {
    "design":    [read_file, write_file, list_directory],
    "implement": [read_file, write_file, append_file, list_directory],
    "test":      [read_file, write_file, append_file, list_directory, run_command],
    "review":    [read_file, write_file],
    "integrate": [read_file, write_file, append_file, list_directory, run_command],
    "configure": [read_file, write_file, list_directory, file_exists],
}

_DEFAULT_TOOLS = [read_file, write_file, list_directory]


def _parse_json_field(val) -> list:
    if isinstance(val, str):
        try:
            return json.loads(val)
        except Exception:
            return []
    return val or []


def _build_system_prompt(task: dict, dep_outputs: list[dict]) -> str:
    """
    Build a task-specific system prompt.

    dep_outputs — results from completed dependency tasks — are injected as
    "PRIOR TASK OUTPUTS" so the agent has memory of what was already built,
    without needing a shared checkpointer across tasks.
    """
    criteria_lines = "\n".join(
        f"  - {c}" for c in _parse_json_field(task.get("acceptance_criteria"))
    )
    output_files = "\n".join(
        f"  - {f}" for f in _parse_json_field(task.get("output_files"))
    )

    prior_context = ""
    if dep_outputs:
        parts = [
            f"[{dep['id']}] {dep['title']}\n{dep['result'] or '(no output recorded)'}"
            for dep in dep_outputs
        ]
        prior_context = "\n\nPRIOR TASK OUTPUTS (from your dependencies):\n" + "\n\n".join(parts)

    return f"""You are an expert software engineer executing a single well-defined task.
Be thorough and complete. Always write all files to disk before finishing.

TASK ID:   {task['id']}
TASK TYPE: {task['task_type']}
TITLE:     {task['title']}

DESCRIPTION:
{task['description']}

FILES TO PRODUCE:
{output_files or '  (none specified)'}

ACCEPTANCE CRITERIA (your output must satisfy ALL of these):
{criteria_lines or '  (none specified)'}{prior_context}

When done, summarise what you implemented in 3-5 bullet points.
Do NOT leave any implementation incomplete."""


# ------------------------------------------------------------------
# LLM-as-judge
# ------------------------------------------------------------------

class _JudgeVerdict(BaseModel):
    passed: bool
    score: int    # 0-10
    reason: str   # one sentence


_JUDGE_SYSTEM_PROMPT = """\
You are a code review judge. Given a task description, its acceptance criteria,
and the AI agent's output, decide whether the task was completed satisfactorily.

Score 0-10:
  8-10 → passed (all acceptance criteria met)
  5-7  → borderline (minor gaps, still usable)
  0-4  → failed (criteria not met, re-execution needed)

Set passed=true if score >= 6. One short sentence for reason."""


async def _judge_task(task: dict, agent_output: str) -> _JudgeVerdict:
    """
    Lightweight LLM-as-judge. Uses judge_model (cheaper) from config.
    Returns a structured verdict with passed/score/reason.
    """
    provider    = config["llm"]["provider"]
    judge_model = config.get("llm", {}).get("judge_model", config["llm"]["model"])

    llm = init_chat_model(f"{provider}:{judge_model}", temperature=0, max_tokens=1000)
    judge_agent = create_agent(
        llm,
        tools=[],
        system_prompt=_JUDGE_SYSTEM_PROMPT,
        response_format=_JudgeVerdict,
    )

    criteria = _parse_json_field(task.get("acceptance_criteria"))
    user_message = f"""TASK: {task['title']}
DESCRIPTION: {task['description']}
ACCEPTANCE CRITERIA: {json.dumps(criteria, indent=2)}

AGENT OUTPUT:
{agent_output[:2000]}"""

    result = await judge_agent.ainvoke({"messages": [{"role": "user", "content": user_message}]})
    return result["structured_response"]


# ------------------------------------------------------------------
# Main entry point — called by orchestrator._execute()
# ------------------------------------------------------------------

async def run_subtask_agent(task: dict, dep_outputs: list[dict] | None = None) -> str:
    """
    Build a fresh agent for a single task and invoke it.

    After the agent returns, an LLM judge verifies the output against
    acceptance_criteria. If it fails (score < 6), raises ValueError so
    the orchestrator's existing retry logic kicks in automatically.
    """
    provider = config["llm"]["provider"]
    model    = config["llm"]["model"]

    llm = init_chat_model(f"{provider}:{model}", temperature=0, max_tokens=3000)

    tools         = _TOOLS_BY_TYPE.get(task.get("task_type", ""), _DEFAULT_TOOLS)
    system_prompt = _build_system_prompt(task, dep_outputs or [])

    logger.info(f"Building agent for task {task['id']} (type={task['task_type']}, tools={[t.name for t in tools]})")

    agent = create_agent(llm, tools=tools, system_prompt=system_prompt)

    # Be explicit that the project may be empty and files must be created from scratch.
    # Without this, the agent explores the empty directory and returns nothing.
    user_message = f"""{task['description']}

The project directory may be empty — there is no existing code to read.
You must CREATE all output files from scratch using write_file.
Do not spend time listing directories. Go directly to writing the output files."""

    final_state = None
    async for step in agent.astream(
        {"messages": [{"role": "user", "content": user_message}]},
        stream_mode="values",
    ):
        last_msg   = step["messages"][-1]
        kind       = type(last_msg).__name__
        tool_calls = getattr(last_msg, "tool_calls", None)
        if tool_calls:
            logger.info(f"Task {task['id']} → {kind} calling tools: {[tc['name'] for tc in tool_calls]}")
        else:
            preview = str(getattr(last_msg, "content", ""))[:200]
            logger.info(f"Task {task['id']} → {kind}: {preview}")
        final_state = step

    # Temporary debug — log full structure of last message to diagnose empty content
    last = final_state["messages"][-1]
    logger.info(f"DEBUG last message type: {type(last).__name__}")
    logger.info(f"DEBUG content: {repr(getattr(last, 'content', 'NO_CONTENT_FIELD'))}")
    logger.info(f"DEBUG tool_calls: {getattr(last, 'tool_calls', 'NO_TOOL_CALLS_FIELD')}")
    logger.info(f"DEBUG additional_kwargs: {getattr(last, 'additional_kwargs', {})}")
    logger.info(f"DEBUG response_metadata: {getattr(last, 'response_metadata', {})}")

    # Walk backwards through messages to find the last non-empty AIMessage content.
    # The final message can be an empty AIMessage if the model finishes a tool loop
    # without producing a text summary — this finds the actual last meaningful output.
    output = ""
    for msg in reversed(final_state["messages"]):
        if type(msg).__name__ == "AIMessage":
            content = getattr(msg, "content", "")
            # Claude returns content as a list of blocks; OpenAI returns a plain string.
            if isinstance(content, list):
                content = " ".join(
                    b.get("text", "") for b in content if isinstance(b, dict)
                )
            if content.strip():
                output = content
                break

    if not output.strip():
        raise ValueError("Agent returned empty output — it did not write any files or produce a summary")

    logger.info(f"Task {task['id']} agent returned {len(output)} chars")

    # ── LLM-as-judge ─────────────────────────────────────────────────
    verdict = await _judge_task(task, output)
    logger.info(
        f"Judge verdict for {task['id']}: score={verdict.score}, "
        f"passed={verdict.passed}, reason={verdict.reason}"
    )
    if not verdict.passed:
        raise ValueError(f"Judge rejected output (score={verdict.score}/10): {verdict.reason}")

    return output