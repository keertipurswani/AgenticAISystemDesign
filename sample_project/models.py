"""
models.py — Data models for the task management app.
Uses Python dataclasses (no ORM, keeping it simple for demo purposes).
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional


class Priority(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class Status(Enum):
    TODO = "todo"
    IN_PROGRESS = "in_progress"
    DONE = "done"


@dataclass
class User:
    id: int
    username: str
    email: str
    created_at: datetime = field(default_factory=datetime.utcnow)
    is_active: bool = True


@dataclass
class Task:
    id: int
    title: str
    description: str
    owner_id: int                          # references User.id
    priority: Priority = Priority.MEDIUM
    status: Status = Status.TODO
    due_date: Optional[datetime] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    tags: list[str] = field(default_factory=list)

    def is_overdue(self) -> bool:
        """Return True if task is past due and not yet done."""
        if self.due_date is None:
            return False
        return datetime.utcnow() > self.due_date and self.status != Status.DONE
