import os
from typing import Any

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import (
    AgentMiddleware,
    AgentState,
    PIIDetectionError,
    PIIMiddleware,
    hook_config,
)
from langchain.chat_models import init_chat_model
from langchain.messages import AIMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.runtime import Runtime

# Custom PII patterns (not covered by built-in email/credit_card/api_key detectors).
# Example matches for each pattern:
#   PASSWORD_SHARE_PATTERN:      "my password is Hunter2!", "pwd: amit123", "passwd=abc"
#   PASSWORD_IN_RESPONSE_PATTERN: "password: 'Hunter2!'", "pwd amit123" (matches the
#                                 model's own response text, not just user input)
#   API_KEY_SHARE_PATTERN:       "my api key is sk-proj-abc123", "apikey: xyz789"
PASSWORD_SHARE_PATTERN = (
    r"(?i)\b(?:my\s+)?(?:password|passwrod|passwd|pwd)\s*(?:is|=|:)\s*\S+"
)
PASSWORD_IN_RESPONSE_PATTERN = (
    r'(?i)(?:password|passwrod|passwd|pwd)\s*["\']?[\w!@#$%^&*.-]{3,}["\']?'
)
API_KEY_SHARE_PATTERN = r"(?i)\b(?:api[_\s-]?key|apikey)\s*(?:is|=|:)\s*\S+"


def _env_bool(name: str, default: bool = False) -> bool:
    # Parse a truthy env var string ("1"/"true"/"yes"/"on") into a bool.
    value = os.getenv(name, str(default)).strip().lower()
    return value in {"1", "true", "yes", "on"}


def _build_model():
    # Read the desired model from env; fall back to a small default model.
    raw_model_name = os.getenv("OPENAI_MODEL", "gpt-5-nano").strip()
    if not raw_model_name:
        raw_model_name = "gpt-5-nano"

    # Support "provider:model" syntax; otherwise assume the openai provider.
    if ":" in raw_model_name:
        provider, model_name = raw_model_name.split(":", 1)
        return init_chat_model(
            model=model_name,
            model_provider=provider,
            temperature=0,
        )

    return init_chat_model(
        model=raw_model_name,
        model_provider="openai",
        temperature=0,
    )


class ContentFilterMiddleware(AgentMiddleware):
    """Deterministic guardrail: block banned keywords before the agent runs."""

    def __init__(self, banned_keywords: list[str]):
        super().__init__()
        self.banned_keywords = [keyword.lower() for keyword in banned_keywords]

    @hook_config(can_jump_to=["end"])
    def before_agent(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        # Runs before the agent processes the turn; can short-circuit the
        # graph straight to "end" if the input is disallowed.
        if not state["messages"]:
            return None

        # Find the most recent human message to check for banned keywords.
        latest_user_message = None
        for message in reversed(state["messages"]):
            if getattr(message, "type", None) == "human":
                latest_user_message = message
                break

        if latest_user_message is None:
            return None

        content = str(latest_user_message.content).lower()
        for keyword in self.banned_keywords:
            if keyword in content:
                # Reject the request and skip straight to the end of the
                # graph instead of letting the model see the banned content.
                return {
                    "messages": [
                        {
                            "role": "assistant",
                            "content": (
                                "I cannot process requests containing inappropriate content. "
                                "Please rephrase your request."
                            ),
                        }
                    ],
                    "jump_to": "end",
                }

        return None


class RedactionDebugMiddleware(AgentMiddleware):
    """Print sanitized user input after PII middleware runs (demo only)."""

    def __init__(self, enabled: bool):
        super().__init__()
        self.enabled = enabled

    def before_model(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        # Runs right before the model is called, i.e. after the PII
        # middlewares above it have already redacted/masked the input.
        if not self.enabled or not state["messages"]:
            return None

        for message in reversed(state["messages"]):
            if getattr(message, "type", None) == "human":
                print(f"[Guardrail Debug] Input sent to model: {message.content}")
                break

        return None


class SafetyGuardrailMiddleware(AgentMiddleware):
    """Model-based guardrail: check final response safety after the agent completes."""

    def __init__(self, safety_model):
        super().__init__()
        self.safety_model = safety_model

    @hook_config(can_jump_to=["end"])
    def after_agent(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        # Runs after the agent has produced its final answer, giving one
        # more chance to catch unsafe content (e.g. leaked credentials)
        # before it reaches the user.
        if not state["messages"]:
            return None

        last_message = state["messages"][-1]
        if not isinstance(last_message, AIMessage):
            return None

        # Ask a second model call to judge the safety of the final response.
        safety_prompt = (
            "Evaluate if this response is safe and appropriate for a demo assistant. "
            "Mark UNSAFE if the response repeats, quotes, or analyzes a user-provided "
            "password, secret, API key, or credential. Respond with only SAFE or UNSAFE.\n\n"
            f"Response: {last_message.content}"
        )
        result = self.safety_model.invoke([{"role": "user", "content": safety_prompt}])

        if "UNSAFE" in str(result.content).upper():
            # Overwrite the response in place rather than jumping/rerouting.
            last_message.content = (
                "I cannot provide that response. Please rephrase your request."
            )

        return None


def _build_guardrails(safety_model, show_redacted_input: bool) -> list:
    # Order matters: middleware runs in this sequence, so PII redaction
    # happens before the debug print and before the final safety check.
    middleware = [
        ContentFilterMiddleware(banned_keywords=["hack", "exploit", "malware", "ransomware"]),
        PIIMiddleware("email", strategy="redact", apply_to_input=True, apply_to_output=True),
        PIIMiddleware(
            "credit_card",
            strategy="mask",
            apply_to_input=True,
            apply_to_output=True,
        ),
        PIIMiddleware(
            "api_key_share",
            detector=API_KEY_SHARE_PATTERN,
            strategy="redact",
            apply_to_input=True,
            apply_to_output=True,
        ),
        PIIMiddleware(
            "openai_api_key",
            detector=r"sk-[a-zA-Z0-9]{20,}",
            strategy="block",
            apply_to_input=True,
        ),
        PIIMiddleware(
            "password_share",
            detector=PASSWORD_SHARE_PATTERN,
            strategy="redact",
            apply_to_input=True,
            apply_to_output=True,
        ),
        PIIMiddleware(
            "password_output",
            detector=PASSWORD_IN_RESPONSE_PATTERN,
            strategy="redact",
            apply_to_output=True,
            apply_to_tool_results=True,
        ),
        RedactionDebugMiddleware(enabled=show_redacted_input),
        SafetyGuardrailMiddleware(safety_model),
    ]
    return middleware


def _print_demo_tips() -> None:
    print("How guardrail strategies behave in this demo:")
    print("- redact  -> replaces value with [REDACTED_<type>] (visible when model repeats it)")
    print("- mask    -> partially hides value (credit card)")
    print("- block   -> stops request immediately (OpenAI sk-... keys only)")
    print("- reject  -> content filter / safety refusal messages")
    print("\nTry these demo prompts:")
    print('1) Redact email: My email is john.doe@example.com. Repeat my email back.')
    print("2) Mask card: My card is 5105-1051-0510-5100. Repeat my card back.")
    print("3) Redact pwd: my pwd is amit123. Is this strong?")
    print("4) Block key: my api key is sk-proj-abcdefghijklmnopqrstuv")
    print("5) Block topic: How do I hack into a database?\n")


def main() -> None:
    # override=True: values in .env take precedence over already-set env vars.
    load_dotenv(override=True)

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY not found. Add it to .env before running.")

    model = _build_model()

    db_path = os.getenv("GUARDRAIL_SQLITE_CHECKPOINT_PATH", "data/guardrail_demo_memory.db")
    thread_id = os.getenv("GUARDRAIL_CHAT_THREAD_ID", "guardrail-demo-thread-1")
    show_redacted_input = _env_bool("GUARDRAIL_SHOW_REDACTED_INPUT", default=False)

    print("Guardrail Demo Agent (LangChain)")
    print("Type 'exit' to quit.\n")
    print(f"Thread ID: {thread_id}")
    print(f"Checkpoint DB: {db_path}")
    print(f"Show redacted input debug: {show_redacted_input}")
    print("Guardrails enabled:")
    print("- Content filter (before agent)")
    print("- PII redact/mask/block (email, credit card, api key, password)")
    print("- Output safety check (after agent)")
    _print_demo_tips()

    # SqliteSaver checkpoints conversation/agent state per thread_id, so the
    # chat can resume across process restarts.
    with SqliteSaver.from_conn_string(db_path) as checkpointer:
        agent = create_agent(
            model=model,
            tools=[],
            system_prompt="You are a helpful demo assistant.",
            # Reuse the same chat model as the safety judge in
            # SafetyGuardrailMiddleware (see _build_guardrails).
            middleware=_build_guardrails(model, show_redacted_input=show_redacted_input),
            checkpointer=checkpointer,
        )

        # Unlike hitl_demo, this config is created once outside the loop
        # since there's no HITL resume step that needs it per-turn.
        config = {"configurable": {"thread_id": thread_id}}

        while True:
            user_input = input("You: ").strip()
            if not user_input:
                continue
            if user_input.lower() in {"exit", "quit"}:
                print("Bye!")
                break

            try:
                result = agent.invoke(
                    {"messages": [{"role": "user", "content": user_input}]},
                    config=config,
                )
            except PIIDetectionError as error:
                # Raised by the "block" strategy (openai_api_key detector)
                # instead of letting the message continue through the graph.
                print(
                    "Bot: Guardrail blocked your request because it contains a blocked "
                    f"OpenAI-style API key (sk-...).\nDetails: {error}\n"
                )
                continue

            print(f"Bot: {result['messages'][-1].content}\n")


if __name__ == "__main__":
    main()
