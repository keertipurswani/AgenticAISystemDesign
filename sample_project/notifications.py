"""
notifications.py — Sends email and Slack alerts on task lifecycle events.
Called by TaskService after every state-changing operation.
"""

from models import Task, User
from integrations.email_client import EmailClient, EmailMessage
from integrations.slack_webhook import SlackWebhook


class NotificationService:
    """Dispatches email + Slack notifications for task events."""

    def __init__(self, email_client: EmailClient, slack: SlackWebhook):
        self.email = email_client
        self.slack = slack

    def on_task_created(self, task: Task, owner: User) -> None:
        self.email.send(EmailMessage(
            to_address=owner.email,
            subject=f"Task created: {task.title}",
            body=(
                f"Hi {owner.username},\n\n"
                f"Your task '{task.title}' has been created "
                f"with priority '{task.priority.value}'.\n"
            ),
        ))
        self.slack.post(
            f":memo: New task by *{owner.username}*: {task.title} "
            f"[{task.priority.value}]"
        )

    def on_task_completed(self, task: Task, owner: User) -> None:
        self.email.send(EmailMessage(
            to_address=owner.email,
            subject=f"Task completed: {task.title}",
            body=(
                f"Hi {owner.username},\n\n"
                f"Great work! Task '{task.title}' has been marked as done.\n"
            ),
        ))
        self.slack.post(
            f":white_check_mark: Task completed by *{owner.username}*: {task.title}"
        )

    def on_task_deleted(self, task: Task, owner: User) -> None:
        self.slack.post(
            f":wastebasket: Task deleted by *{owner.username}*: {task.title}"
        )

    def on_task_assigned(self, task: Task, assignee: User, assigner: User) -> None:
        self.email.send(EmailMessage(
            to_address=assignee.email,
            subject=f"Task assigned to you: {task.title}",
            body=(
                f"Hi {assignee.username},\n\n"
                f"{assigner.username} assigned you the task '{task.title}'.\n"
            ),
        ))
        self.slack.post(
            f":incoming_envelope: *{assigner.username}* assigned "
            f"'{task.title}' to *{assignee.username}*"
        )
