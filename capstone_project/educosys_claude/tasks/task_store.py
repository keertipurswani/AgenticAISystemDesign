from __future__ import annotations

import json
import sqlite3
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from educosys_claude.observability.logger import get_logger

logger = get_logger(__name__)


class TaskStatus(str, Enum):
    """
    All valid states a task can be in.

    State machine:
        PENDING ──(claim)──► IN_PROGRESS ──(success)──► COMPLETED  (terminal)
                                    │
                                    ├──(failure, retries left)──► PENDING
                                    ├──(failure, no retries)────► FAILED
                                    └──(dep failed)─────────────► BLOCKED
        SKIPPED  (terminal — human explicitly skipped)
    """
    PENDING     = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED   = "completed"
    FAILED      = "failed"
    BLOCKED     = "blocked"
    SKIPPED     = "skipped"


class TaskType(str, Enum):
    DESIGN    = "design"
    IMPLEMENT = "implement"
    TEST      = "test"
    REVIEW    = "review"
    INTEGRATE = "integrate"
    CONFIGURE = "configure"


@dataclass
class Task:
    """
    Single unit of work inside a project.

    depends_on and output_files are stored as JSON strings in SQLite
    and converted to/from lists at the Python boundary.
    """
    id: str
    project_id: str
    title: str
    description: str
    task_type: str
    status: str               = TaskStatus.PENDING
    depends_on: str           = "[]"   # JSON list of task IDs that must complete first
    output_files: str         = "[]"   # JSON list of file paths this task will produce
    result: str | None        = None   # agent's summary output on success
    error: str | None         = None   # error message on failure or crash
    retry_count: int          = 0
    max_retries: int          = 3
    execution_order: int      = 0      # position in plan — used for display ordering
    created_at: float         = field(default_factory=time.time)
    started_at: float | None  = None
    completed_at: float | None = None


class SQLiteTaskStore:
    """
    Authoritative persistence layer for all task state.

    Design rules:
    - WAL mode: crash-safe, allows concurrent reads while the orchestrator writes
    - Every connection is opened and closed per operation — no shared long-lived connection
    - Every state transition is a single atomic UPDATE — write to DB before acting on state
    - Execution is single-process and serial, so any task left IN_PROGRESS when the
      process starts up again is unconditionally orphaned (no heartbeat needed)

    DB layout:
        projects table  — one row per /plan invocation
        tasks table     — 10-40 rows per project, one per subtask
    """

    def __init__(self, db_path: str = ".educosys/tasks.db") -> None:
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._setup_db()
        logger.info(f"SQLiteTaskStore initialised at {db_path}")

    @contextmanager
    def _conn(self):
        """
        Open a fresh connection, enable WAL + foreign keys, yield, commit, close.
        Using a context manager per operation avoids shared-connection threading issues.
        """
        conn = sqlite3.connect(self.db_path, timeout=10)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _setup_db(self) -> None:
        with self._conn() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS projects (
                    id          TEXT PRIMARY KEY,
                    name        TEXT NOT NULL,
                    goal        TEXT NOT NULL,
                    plan_json   TEXT,
                    status      TEXT DEFAULT 'planning',
                    created_at  REAL DEFAULT (unixepoch()),
                    approved_at REAL
                );

                CREATE TABLE IF NOT EXISTS tasks (
                    id               TEXT PRIMARY KEY,
                    project_id       TEXT REFERENCES projects(id),
                    title            TEXT NOT NULL,
                    description      TEXT NOT NULL,
                    task_type        TEXT NOT NULL,
                    status           TEXT DEFAULT 'pending',
                    depends_on           TEXT DEFAULT '[]',
                    output_files         TEXT DEFAULT '[]',
                    acceptance_criteria  TEXT DEFAULT '[]',
                    result           TEXT,
                    error            TEXT,
                    retry_count      INTEGER DEFAULT 0,
                    max_retries      INTEGER DEFAULT 3,
                    execution_order  INTEGER DEFAULT 0,
                    created_at       REAL DEFAULT (unixepoch()),
                    started_at       REAL,
                    completed_at     REAL
                );

                CREATE INDEX IF NOT EXISTS idx_tasks_project
                    ON tasks(project_id, status);
            """)

    # ------------------------------------------------------------------
    # Project operations
    # ------------------------------------------------------------------

    def create_project(self, goal: str, plan) -> str:
        """
        Persist an approved ExecutionPlan as a project + task rows.
        Returns the new project_id (UUID).
        """
        project_id = str(uuid.uuid4())
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO projects(id, name, goal, plan_json, status) VALUES (?,?,?,?,?)",
                (project_id, plan.project_name, goal, plan.model_dump_json(), "approved")
            )
            for i, pt in enumerate(plan.tasks):
                conn.execute(
                    """INSERT INTO tasks
                       (id, project_id, title, description, task_type,
                        depends_on, output_files, acceptance_criteria, execution_order)
                       VALUES (?,?,?,?,?,?,?,?,?)""",
                    (
                        pt.id, project_id, pt.title, pt.description,
                        pt.task_type.value,
                        json.dumps(pt.depends_on),
                        json.dumps(pt.output_files),
                        json.dumps(pt.acceptance_criteria),
                        i,
                    )
                )
        logger.info(f"Project {project_id} created with {len(plan.tasks)} tasks")
        return project_id

    def get_latest_approved_project(self) -> str | None:
        """Return the most recently approved project_id, or None if none exists."""
        with self._conn() as conn:
            row = conn.execute(
                "SELECT id FROM projects WHERE status='approved' ORDER BY created_at DESC LIMIT 1"
            ).fetchone()
        return row["id"] if row else None

    # ------------------------------------------------------------------
    # Atomic state transitions
    # ------------------------------------------------------------------

    def claim_task(self, task_id: str) -> bool:
        """
        Atomically transition a task from PENDING → IN_PROGRESS.
        Returns True if the claim succeeded, False if already claimed.
        """
        with self._conn() as conn:
            cur = conn.execute(
                """UPDATE tasks
                   SET status     = 'in_progress',
                       started_at = unixepoch()
                   WHERE id = ? AND status = 'pending'""",
                (task_id,)
            )
        return cur.rowcount == 1

    def complete_task(self, task_id: str, result: str) -> None:
        with self._conn() as conn:
            conn.execute(
                """UPDATE tasks
                   SET status       = 'completed',
                       result       = ?,
                       completed_at = unixepoch()
                   WHERE id = ?""",
                (result, task_id)
            )
        logger.info(f"Task {task_id} completed")

    def fail_task(self, task_id: str, error: str) -> None:
        """
        Record failure and apply retry logic in a single atomic UPDATE.
        SQL CASE: retries left → 'pending', exhausted → 'failed'.
        """
        with self._conn() as conn:
            conn.execute(
                """UPDATE tasks
                   SET retry_count = retry_count + 1,
                       error       = ?,
                       status      = CASE
                           WHEN retry_count + 1 < max_retries THEN 'pending'
                           ELSE 'failed'
                       END,
                       started_at  = NULL
                   WHERE id = ?""",
                (error, task_id)
            )
        logger.info(f"Task {task_id} failed: {error[:80]}")

    def block_task(self, task_id: str, reason: str) -> None:
        with self._conn() as conn:
            conn.execute(
                "UPDATE tasks SET status='blocked', error=? WHERE id=?",
                (reason, task_id)
            )

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    # get ready tasks - get tasks that are in pending state and 
    # their dependents are all done
    def get_ready_tasks(self, project_id: str) -> list[dict]:
        """Return PENDING tasks whose every dependency is COMPLETED or SKIPPED."""
        with self._conn() as conn:
            rows = conn.execute(
                """SELECT * FROM tasks
                   WHERE project_id = ? AND status = 'pending'
                   ORDER BY execution_order""",
                (project_id,)
            ).fetchall()

        # list of pending tasks

        ready = []
        for row in rows:
            # for each of the pending tasks
            deps = json.loads(row["depends_on"])
            if self._all_deps_done(project_id, deps):
                ready.append(dict(row))
        return ready

    # checking if dependents of task are done - true or false
    def _all_deps_done(self, project_id: str, dep_ids: list[str]) -> bool:
        if not dep_ids:
            return True
        placeholders = ",".join(["?"] * len(dep_ids))
        with self._conn() as conn:
            row = conn.execute(
                f"""SELECT COUNT(*) FROM tasks
                    WHERE id IN ({placeholders})
                      AND status NOT IN ('completed', 'skipped')""",
                dep_ids
            ).fetchone()
        # no of dependents that are not completed/skipped should be 0
        return row[0] == 0

    def get_all_tasks(self, project_id: str) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM tasks WHERE project_id=? ORDER BY execution_order",
                (project_id,)
            ).fetchall()
        return [dict(r) for r in rows]

    def get_progress(self, project_id: str) -> dict[str, int]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT status, COUNT(*) as n FROM tasks WHERE project_id=? GROUP BY status",
                (project_id,)
            ).fetchall()
        return {r["status"]: r["n"] for r in rows}

    def get_dep_results(self, dep_ids: list[str]) -> list[dict]:
        """
        Fetch title + result of completed dependency tasks.
        Injected into each agent's system prompt so it knows what prior tasks produced.
        """
        if not dep_ids:
            return []
        placeholders = ",".join(["?"] * len(dep_ids))
        with self._conn() as conn:
            rows = conn.execute(
                f"""SELECT id, title, result FROM tasks
                    WHERE id IN ({placeholders})
                      AND status = 'completed'""",
                dep_ids,
            ).fetchall()
        return [dict(r) for r in rows]