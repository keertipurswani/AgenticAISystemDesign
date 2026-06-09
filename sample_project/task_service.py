"""
task_service.py — Business logic layer for task operations.
Sits between the API/CLI layer and the database.
Enforces ownership rules, permission policy, billing limits, audit logging,
and dispatches notifications on every state change.
"""

from database import Database
from models import Task, Priority, Status
from auth import AuthService, AuthError
from permissions import PermissionPolicy
from billing import BillingService
from audit_log import AuditLogger, AuditEventType
from notifications import NotificationService


class TaskService:
    """All task operations go through here so no check is ever skipped."""

    def __init__(
        self,
        db: Database,
        auth: AuthService,
        policy: PermissionPolicy,
        billing: BillingService,
        audit: AuditLogger,
        notifications: NotificationService,
    ):
        self.db = db
        self.auth = auth
        self.policy = policy
        self.billing = billing
        self.audit = audit
        self.notifications = notifications

    def create_task(
        self,
        token: str,
        title: str,
        description: str,
        priority: Priority = Priority.MEDIUM,
        tags: list[str] = None,
    ) -> Task:
        """Create a task owned by the currently logged-in user."""
        user = self.auth.get_current_user(token)
        self.policy.require(user, "task:create")
        self.billing.check_task_limit(user)
        task = self.db.create_task(
            title=title,
            description=description,
            owner_id=user.id,
            priority=priority,
            tags=tags or [],
        )
        self.audit.record(
            AuditEventType.TASK_CREATED,
            actor_id=user.id,
            target_id=task.id,
            detail=f"title='{title}' priority='{priority.value}'",
        )
        self.notifications.on_task_created(task, user)
        return task

    def complete_task(self, token: str, task_id: int) -> Task:
        """Mark a task as done. Only the owner can do this."""
        user = self.auth.get_current_user(token)
        self.policy.require(user, "task:complete")
        task = self.db.get_task(task_id)
        if task is None:
            raise ValueError(f"Task {task_id} not found")
        if task.owner_id != user.id:
            raise AuthError("You can only complete your own tasks")
        updated = self.db.update_task_status(task_id, Status.DONE)
        self.audit.record(
            AuditEventType.TASK_COMPLETED,
            actor_id=user.id,
            target_id=task_id,
            detail=f"task_id={task_id}",
        )
        self.notifications.on_task_completed(updated, user)
        return updated

    def list_my_tasks(self, token: str) -> list[Task]:
        """Return all tasks belonging to the current user."""
        user = self.auth.get_current_user(token)
        self.policy.require(user, "task:view")
        return self.db.list_tasks(owner_id=user.id)

    def delete_task(self, token: str, task_id: int) -> bool:
        """Delete a task. Only the owner can delete it."""
        user = self.auth.get_current_user(token)
        self.policy.require(user, "task:delete")
        task = self.db.get_task(task_id)
        if task is None:
            raise ValueError(f"Task {task_id} not found")
        if task.owner_id != user.id:
            raise AuthError("You can only delete your own tasks")
        deleted = self.db.delete_task(task_id)
        if deleted:
            self.audit.record(
                AuditEventType.TASK_DELETED,
                actor_id=user.id,
                target_id=task_id,
                detail=f"task_id={task_id}",
            )
            self.notifications.on_task_deleted(task, user)
        return deleted

    def assign_task(self, token: str, task_id: int, assignee_id: int) -> Task:
        """
        Reassign a task to another user.
        Requires task:assign permission (admin only).
        Updates owner_id so the new owner passes future ownership checks.
        """
        user = self.auth.get_current_user(token)
        self.policy.require(user, "task:assign")
        task = self.db.get_task(task_id)
        if task is None:
            raise ValueError(f"Task {task_id} not found")
        assignee = self.db.get_user(assignee_id)
        if assignee is None:
            raise ValueError(f"Assignee {assignee_id} not found")
        task.owner_id = assignee_id
        self.audit.record(
            AuditEventType.TASK_ASSIGNED,
            actor_id=user.id,
            target_id=task_id,
            detail=f"assigned to user_id={assignee_id}",
        )
        self.notifications.on_task_assigned(task, assignee, user)
        return task

    def search_tasks(self, token: str, keyword: str) -> list[Task]:
        """Search user's tasks by keyword in title or description."""
        user = self.auth.get_current_user(token)
        self.policy.require(user, "task:view")
        all_matches = self.db.search_tasks(keyword)
        return [t for t in all_matches if t.owner_id == user.id]
