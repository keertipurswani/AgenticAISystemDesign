"""
integrations/slack_webhook.py — Posts messages to a Slack channel via webhook.
Prints to stdout in demo mode (no real HTTP call).
"""


class SlackWebhook:
    """Wraps a Slack incoming webhook URL."""

    def __init__(self, webhook_url: str = "https://hooks.slack.com/demo"):
        self.webhook_url = webhook_url
        self._posted: list[str] = []   # captured messages for testing

    def post(self, text: str) -> None:
        self._posted.append(text)
        print(f"[SLACK] {text}")

    def posted_messages(self) -> list[str]:
        return list(self._posted)
