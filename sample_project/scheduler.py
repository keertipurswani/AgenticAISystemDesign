"""
scheduler.py — Recurring task scheduler.
Allows tasks to be auto-created on a cron-like schedule (daily/weekly).
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Optional
from models import Priority, Task, User
from database import Database
from billing import BillingService
from audit_log import AuditLogger, AuditEventType


class RecurrenceInterval(Enum):
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"


@dataclass
class ScheduledTask:
    id: int
    owner_id: int
    title: str
    description: str
    priority: Priority
    interval: RecurrenceInterval
    next_run: datetime
    tags: list[str] = field(default_factory=list)
    is_active: bool = True


class TaskScheduler:
    """
    Manages recurring task templates.
    On each tick(), due schedules are materialized into real Task records.
    Respects billing limits — skips creation and logs if plan_tier cap is hit.
    """

    def __init__(self, db: Database, billing: BillingService, audit: AuditLogger):
        self.db = db
        self.billing = billing
        self.audit = audit
        self._schedules: dict[int, ScheduledTask] = {}
        self._next_id = 1

    def add_schedule(
        self,
        owner: User,
        title: str,
        description: str,
        priority: Priority,
        interval: RecurrenceInterval,
        first_run: Optional[datetime] = None,
        tags: list[str] = None,
    ) -> ScheduledTask:
        schedule = ScheduledTask(
            id=self._next_id,
            owner_id=owner.id,
            title=title,
            description=description,
            priority=priority,
            interval=interval,
            next_run=first_run or datetime.utcnow(),
            tags=tags or [],
        )
        self._schedules[schedule.id] = schedule
        self._next_id += 1
        return schedule

    def cancel_schedule(self, schedule_id: int) -> None:
        if schedule_id in self._schedules:
            self._schedules[schedule_id].is_active = False

    def tick(self, now: Optional[datetime] = None) -> list[Task]:
        """Materialize all due schedules into Task records. Call periodically."""
        now = now or datetime.utcnow()
        created: list[Task] = []

        for schedule in self._schedules.values():
            if not schedule.is_active or schedule.next_run > now:
                continue

            owner = self.db.get_user(schedule.owner_id)
            if owner is None or not owner.is_active:
                continue

            try:
                self.billing.check_task_limit(owner)
            except Exception:
                self.audit.record(
                    AuditEventType.PERMISSION_DENIED,
                    actor_id=owner.id,
                    detail=f"Scheduled task skipped — plan_tier limit reached for schedule {schedule.id}",
                )
                continue

            task = self.db.create_task(
                title=schedule.title,
                description=schedule.description,
                owner_id=schedule.owner_id,
                priority=schedule.priority,
                tags=schedule.tags,
            )
            self.audit.record(
                AuditEventType.TASK_CREATED,
                actor_id=owner.id,
                target_id=task.id,
                detail=f"Created by scheduler schedule_id={schedule.id}",
            )
            created.append(task)
            schedule.next_run = self._advance(schedule.next_run, schedule.interval)

        return created

    @staticmethod
    def _advance(dt: datetime, interval: RecurrenceInterval) -> datetime:
        if interval == RecurrenceInterval.DAILY:
            return dt + timedelta(days=1)
        if interval == RecurrenceInterval.WEEKLY:
            return dt + timedelta(weeks=1)
        # monthly — approximate with 30 days
        return dt + timedelta(days=30)
