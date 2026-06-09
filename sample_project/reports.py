"""
reports.py — Aggregated statistics across users and tasks.
Requires report:view permission (checked via PermissionPolicy).
"""

from dataclasses import dataclass
from models import Status, Priority
from database import Database
from permissions import PermissionPolicy
from models import User


@dataclass
class UserReport:
    user_id: int
    username: str
    total_tasks: int
    completed_tasks: int
    overdue_tasks: int
    tasks_by_priority: dict[str, int]


@dataclass
class SystemReport:
    total_users: int
    total_tasks: int
    completed_tasks: int
    tasks_by_status: dict[str, int]
    tasks_by_priority: dict[str, int]


class ReportService:
    """Generates reports. Enforces report:view permission before returning data."""

    def __init__(self, db: Database, policy: PermissionPolicy):
        self.db = db
        self.policy = policy

    def user_report(self, requester: User, target_user_id: int) -> UserReport:
        self.policy.require(requester, "report:view")
        target = self.db.get_user(target_user_id)
        if target is None:
            raise ValueError(f"User {target_user_id} not found")
        tasks = self.db.list_tasks(owner_id=target_user_id)
        by_priority: dict[str, int] = {p.value: 0 for p in Priority}
        for t in tasks:
            by_priority[t.priority.value] += 1
        return UserReport(
            user_id=target.id,
            username=target.username,
            total_tasks=len(tasks),
            completed_tasks=sum(1 for t in tasks if t.status == Status.DONE),
            overdue_tasks=sum(1 for t in tasks if t.is_overdue()),
            tasks_by_priority=by_priority,
        )

    def system_report(self, requester: User) -> SystemReport:
        self.policy.require(requester, "report:view")
        all_tasks = self.db.list_tasks()
        all_users = self.db.list_users()
        by_status: dict[str, int] = {s.value: 0 for s in Status}
        by_priority: dict[str, int] = {p.value: 0 for p in Priority}
        for t in all_tasks:
            by_status[t.status.value] += 1
            by_priority[t.priority.value] += 1
        return SystemReport(
            total_users=len(all_users),
            total_tasks=len(all_tasks),
            completed_tasks=by_status.get(Status.DONE.value, 0),
            tasks_by_status=by_status,
            tasks_by_priority=by_priority,
        )
