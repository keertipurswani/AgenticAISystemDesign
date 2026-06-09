"""
database.py — In-memory database layer for the task management app.
In a real app this would use SQLAlchemy or similar. Kept simple for demos.
"""

from typing import Optional
from models import Task, User, Status, Priority


class Database:
    """Simple in-memory store — a dict per table, keyed by id."""

    def __init__(self):
        self._users: dict[int, User] = {}
        self._tasks: dict[int, Task] = {}
        self._next_user_id = 1
        self._next_task_id = 1

    # ------------------------------------------------------------------
    # Users
    # ------------------------------------------------------------------

    def create_user(self, username: str, email: str) -> User:
        user = User(id=self._next_user_id, username=username, email=email)
        self._users[user.id] = user
        self._next_user_id += 1
        return user

    def get_user(self, user_id: int) -> Optional[User]:
        return self._users.get(user_id)

    def get_user_by_email(self, email: str) -> Optional[User]:
        for user in self._users.values():
            if user.email == email:
                return user
        return None

    def list_users(self) -> list[User]:
        return list(self._users.values())

    # ------------------------------------------------------------------
    # Tasks
    # ------------------------------------------------------------------

    def create_task(
        self,
        title: str,
        description: str,
        owner_id: int,
        priority: Priority = Priority.MEDIUM,
        tags: list[str] = None,
    ) -> Task:
        task = Task(
            id=self._next_task_id,
            title=title,
            description=description,
            owner_id=owner_id,
            priority=priority,
            tags=tags or [],
        )
        self._tasks[task.id] = task
        self._next_task_id += 1
        return task

    def get_task(self, task_id: int) -> Optional[Task]:
        return self._tasks.get(task_id)

    def update_task_status(self, task_id: int, status: Status) -> Optional[Task]:
        task = self._tasks.get(task_id)
        if task:
            task.status = status
        return task

    def list_tasks(self, owner_id: Optional[int] = None) -> list[Task]:
        tasks = list(self._tasks.values())
        if owner_id is not None:
            tasks = [t for t in tasks if t.owner_id == owner_id]
        return tasks

    def delete_task(self, task_id: int) -> bool:
        if task_id in self._tasks:
            del self._tasks[task_id]
            return True
        return False

    def search_tasks(self, keyword: str) -> list[Task]:
        """Search tasks by keyword in title or description (case-insensitive)."""
        keyword = keyword.lower()
        return [
            t for t in self._tasks.values()
            if keyword in t.title.lower() or keyword in t.description.lower()
        ]
