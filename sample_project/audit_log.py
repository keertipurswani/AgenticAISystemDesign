"""
audit_log.py — Append-only audit trail for all state-changing operations.
Every create / update / delete action on tasks and users is recorded here.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional


class AuditEventType(Enum):
    TASK_CREATED = "task_created"
    TASK_COMPLETED = "task_completed"
    TASK_DELETED = "task_deleted"
    TASK_ASSIGNED = "task_assigned"
    USER_REGISTERED = "user_registered"
    USER_DEACTIVATED = "user_deactivated"
    LOGIN_SUCCESS = "login_success"
    LOGIN_FAILURE = "login_failure"
    PERMISSION_DENIED = "permission_denied"


@dataclass
class AuditEntry:
    event_type: AuditEventType
    actor_id: Optional[int]          # user who triggered the event
    target_id: Optional[int]         # resource affected (task_id or user_id)
    detail: str
    timestamp: datetime = field(default_factory=datetime.utcnow)


class AuditLogger:
    """In-memory audit log. In production this would write to a DB table."""

    def __init__(self):
        self._log: list[AuditEntry] = []

    def record(
        self,
        event_type: AuditEventType,
        actor_id: Optional[int] = None,
        target_id: Optional[int] = None,
        detail: str = "",
    ) -> AuditEntry:
        entry = AuditEntry(
            event_type=event_type,
            actor_id=actor_id,
            target_id=target_id,
            detail=detail,
        )
        self._log.append(entry)
        return entry

    def get_events_for_user(self, user_id: int) -> list[AuditEntry]:
        return [e for e in self._log if e.actor_id == user_id]

    def get_events_for_target(self, target_id: int) -> list[AuditEntry]:
        return [e for e in self._log if e.target_id == target_id]

    def get_events_by_type(self, event_type: AuditEventType) -> list[AuditEntry]:
        return [e for e in self._log if e.event_type == event_type]

    def all_entries(self) -> list[AuditEntry]:
        return list(self._log)
