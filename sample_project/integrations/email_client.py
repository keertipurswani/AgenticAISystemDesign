"""
integrations/email_client.py — Thin SMTP wrapper.
In production this would use SendGrid / SES. Prints to stdout for demo.
"""

from dataclasses import dataclass


@dataclass
class EmailMessage:
    to_address: str
    subject: str
    body: str


class EmailClient:
    """Sends transactional emails. Swap send() for a real SMTP call in prod."""

    def __init__(self, from_address: str = "noreply@taskapp.dev"):
        self.from_address = from_address
        self._sent: list[EmailMessage] = []   # in-memory outbox for testing

    def send(self, message: EmailMessage) -> None:
        self._sent.append(message)
        print(
            f"[EMAIL] To: {message.to_address} | "
            f"Subject: {message.subject}"
        )

    def sent_messages(self) -> list[EmailMessage]:
        return list(self._sent)
