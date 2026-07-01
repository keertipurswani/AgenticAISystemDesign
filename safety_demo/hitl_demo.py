import os
import json
import subprocess
from pathlib import Path

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware, ToolCallRequest
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langgraph.types import Command
from langgraph.checkpoint.sqlite import SqliteSaver


def _build_model():
    # Read the desired model from env; fall back to a small default model.
    model_name = os.getenv("OPENAI_MODEL", "").strip() or "gpt-5-nano"
    # init_chat_model understands "provider:model" strings, so default to
    # the openai provider when the user only supplied a bare model name.
    if ":" not in model_name:
        model_name = f"openai:{model_name}"
    return init_chat_model(model_name, temperature=0)


def _build_tools(workspace_root: Path) -> list:
    def _safe_resolve(user_path: str) -> Path:
        # Resolve the path and make sure it can't escape the workspace
        # (blocks things like "../../etc/passwd").
        candidate = (workspace_root / user_path).resolve()
        try:
            candidate.relative_to(workspace_root)
        except ValueError as exc:
            raise ValueError("Path must stay inside workspace root.") from exc
        return candidate

    @tool
    def list_files(path: str = ".") -> str:
        """List files and folders for a directory path within the workspace."""
        target = _safe_resolve(path)
        if not target.exists():
            return f"Path does not exist: {path}"
        if not target.is_dir():
            return f"Path is not a directory: {path}"

        # Sort directories first, then files, alphabetically (case-insensitive).
        entries = sorted(target.iterdir(), key=lambda p: (p.is_file(), p.name.lower()))
        if not entries:
            return f"Directory is empty: {path}"

        lines = []
        for entry in entries:
            rel = entry.relative_to(workspace_root).as_posix()
            kind = "dir" if entry.is_dir() else "file"
            lines.append(f"[{kind}] {rel}")
        return "\n".join(lines)

    @tool
    def read_file(path: str) -> str:
        """Read and return the full content of a UTF-8 text file."""
        target = _safe_resolve(path)
        if not target.exists():
            return f"File does not exist: {path}"
        if not target.is_file():
            return f"Path is not a file: {path}"
        return target.read_text(encoding="utf-8")

    @tool
    def write_file(path: str, content: str) -> str:
        """Create or overwrite a UTF-8 text file with provided content."""
        target = _safe_resolve(path)
        # Create any missing parent directories so writes never fail on that.
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        rel = target.relative_to(workspace_root).as_posix()
        return f"Wrote file: {rel}"

    @tool
    def append_file(path: str, content: str) -> str:
        """Append content to a UTF-8 text file, creating it if missing."""
        target = _safe_resolve(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("a", encoding="utf-8") as handle:
            handle.write(content)
        rel = target.relative_to(workspace_root).as_posix()
        return f"Appended to file: {rel}"

    @tool
    def update_file(path: str, old_text: str, new_text: str, replace_all: bool = False) -> str:
        """Update text in a file by replacing old_text with new_text."""
        target = _safe_resolve(path)
        if not target.exists():
            return f"File does not exist: {path}"
        if not target.is_file():
            return f"Path is not a file: {path}"

        content = target.read_text(encoding="utf-8")
        if old_text not in content:
            return "old_text was not found in file."

        if replace_all:
            updated = content.replace(old_text, new_text)
            count = content.count(old_text)
        else:
            # Only replace the first occurrence.
            updated = content.replace(old_text, new_text, 1)
            count = 1

        target.write_text(updated, encoding="utf-8")
        rel = target.relative_to(workspace_root).as_posix()
        return f"Updated file: {rel} (replacements: {count})"

    @tool
    def delete_file(path: str) -> str:
        """Delete a file at a given path within the workspace."""
        target = _safe_resolve(path)
        if not target.exists():
            return f"File does not exist: {path}"
        if not target.is_file():
            return f"Path is not a file: {path}"
        target.unlink()
        rel = target.relative_to(workspace_root).as_posix()
        return f"Deleted file: {rel}"

    @tool
    def run_terminal_command(command: str, timeout_seconds: int = 30) -> str:
        """Run a terminal command in workspace root and return output."""
        result = subprocess.run(
            command,
            cwd=workspace_root,
            shell=True,
            capture_output=True,
            text=True,
            # Clamp the timeout to a sane range (1s to 2 minutes).
            timeout=max(1, min(timeout_seconds, 120)),
        )
        stdout = result.stdout.strip()
        stderr = result.stderr.strip()
        parts = [f"exit_code: {result.returncode}"]
        if stdout:
            parts.append(f"stdout:\n{stdout}")
        if stderr:
            parts.append(f"stderr:\n{stderr}")
        return "\n\n".join(parts)

    @tool
    def ask_user(question: str) -> str:
        """Ask the human user for clarification or approval-related input."""
        # The real answer comes back later through the HITL interrupt/resume
        # flow; this return value is just a placeholder for the tool call.
        return f"User input placeholder for question: {question}"

    return [
        list_files,
        read_file,
        write_file,
        append_file,
        update_file,
        delete_file,
        run_terminal_command,
        ask_user,
    ]


def _is_dangerous_command(request: ToolCallRequest) -> bool:
    # Used as the "when" predicate for run_terminal_command's HITL rule:
    # only pause for approval if the command looks destructive.
    command = str(request.tool_call.get("args", {}).get("command", "")).lower()
    dangerous_tokens = [
        "rm ",
        "del ",
        "rmdir ",
        "format ",
        "shutdown",
        "reboot",
        "mkfs",
        "diskpart",
        "reg delete",
        "powershell -encodedcommand",
        "drop table",
        "truncate table",
    ]
    return any(token in command for token in dangerous_tokens)


def _extract_state(result):
    # agent.invoke() can return either the raw state dict or an object
    # with a .value attribute depending on context; normalize to the dict.
    return getattr(result, "value", result)


def _print_agent_response(result) -> None:
    state = _extract_state(result)
    print(f"Bot: {state['messages'][-1].content}\n")


def _handle_interrupts(agent, result, config: dict) -> dict:
    # HumanInTheLoopMiddleware pauses the graph (an "interrupt") whenever a
    # gated tool call needs a human decision. Keep resolving interrupts and
    # resuming the agent until it finishes without asking for anything else.
    while getattr(result, "interrupts", None):
        interrupt = result.interrupts[0]
        payload = interrupt.value
        action_requests = payload.get("action_requests", [])
        review_configs = payload.get("review_configs", [])

        if not action_requests:
            break

        decisions = []
        for index, action in enumerate(action_requests):
            tool_name = action.get("name", "unknown_tool")
            arguments = action.get("args", {})
            allowed = review_configs[index].get(
                "allowed_decisions",
                ["approve", "edit", "reject", "respond"],
            )

            print(f"\nApproval needed for tool: {tool_name}")
            print(f"Arguments: {arguments}")
            print(f"Allowed decisions: {', '.join(allowed)}")

            # ask_user calls expect a free-text answer rather than an
            # approve/edit/reject decision.
            if "respond" in allowed and tool_name == "ask_user":
                user_reply = input("Your response: ").strip()
                decisions.append({"type": "respond", "message": user_reply})
                continue

            choice = input("Reason : ").strip().lower()
            while choice not in allowed:
                choice = input(f"Choose one of {allowed}: ").strip().lower()

            if choice == "approve":
                decisions.append({"type": "approve"})
            elif choice == "reject":
                feedback = input("Rejection feedback (optional): ").strip()
                decision = {"type": "reject"}
                if feedback:
                    decision["message"] = feedback
                decisions.append(decision)
            elif choice == "edit":
                print("Provide edited args as JSON object. Example: {\"path\":\"notes.txt\"}")
                edited_args_raw = input("Edited args JSON: ").strip()
                # Reuse the original arguments if the user submits nothing.
                edited_args = json.loads(edited_args_raw) if edited_args_raw else arguments
                decisions.append(
                    {
                        "type": "edit",
                        "edited_action": {"name": tool_name, "args": edited_args},
                    }
                )

        # Resume the graph from where it paused, supplying the human's
        # decisions for each pending tool call. This may trigger another
        # interrupt (loop continues) or return a final response.
        result = agent.invoke(
            Command(resume={"decisions": decisions}),
            config=config,
            version="v2",
        )
    return result


def main() -> None:
    # override=True: values in .env take precedence over already-set env vars.
    load_dotenv(override=True)

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY not found. Add it to .env before running.")

    model = _build_model()
    # Tools are sandboxed to the repo root (one level above this file).
    workspace_root = Path(__file__).resolve().parents[1]
    tools = _build_tools(workspace_root)

    db_path = os.getenv("SQLITE_CHECKPOINT_PATH", "data/chatbot_memory.db")
    # Make sure the checkpoint DB's parent directory exists before connecting.
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    thread_id = os.getenv("CHAT_THREAD_ID", "demo-thread-1")

    print("Simple Agentic Chatbot (LangChain)")
    print("Type 'exit' to quit.\n")
    print(f"Thread ID: {thread_id}")
    print(f"Checkpoint DB: {db_path}\n")

    # SqliteSaver checkpoints conversation/agent state per thread_id, so the
    # chat can resume across process restarts.
    with SqliteSaver.from_conn_string(db_path) as checkpointer:
        # Gate specific tool calls behind human approval before they execute.
        hitl_middleware = HumanInTheLoopMiddleware(
            interrupt_on={
                "delete_file": {"allowed_decisions": ["approve", "reject"]},
                "write_file": {"allowed_decisions": ["approve", "edit", "reject"]},
                "append_file": {"allowed_decisions": ["approve", "edit", "reject"]},
                "update_file": {"allowed_decisions": ["approve", "edit", "reject"]},
                "run_terminal_command": {
                    "allowed_decisions": ["approve", "edit", "reject"],
                    # Only interrupt for commands that look destructive.
                    "when": _is_dangerous_command,
                },
                "ask_user": {"allowed_decisions": ["respond"]},
            },
            description_prefix="Tool execution pending approval",
        )
        agent = create_agent(
            model=model,
            tools=tools,
            system_prompt=(
                "You are a helpful demo assistant with file and terminal tools. "
                "Use tools when needed, explain actions briefly, and avoid destructive "
                "commands unless explicitly requested. If you need clarification from "
                "the human, call the ask_user tool."
            ),
            middleware=[hitl_middleware],
            checkpointer=checkpointer,
        )

        while True:
            user_input = input("You: ").strip()
            if not user_input:
                continue
            if user_input.lower() in {"exit", "quit"}:
                print("Bye!")
                break

            # thread_id ties this turn to the persisted conversation state.
            config = {"configurable": {"thread_id": thread_id}}
            result = agent.invoke(
                {"messages": [("user", user_input)]},
                config=config,
                version="v2",
            )
            # Resolve any HITL approval prompts before printing the reply.
            result = _handle_interrupts(agent, result, config)
            _print_agent_response(result)


if __name__ == "__main__":
    main()
