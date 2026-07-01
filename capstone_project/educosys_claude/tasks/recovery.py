from __future__ import annotations

from rich.console import Console

from educosys_claude.tasks.task_store import SQLiteTaskStore
from educosys_claude.observability.logger import get_logger

logger = get_logger(__name__)
console = Console()


class RecoveryManager:
    """
    Detects and resets tasks left IN_PROGRESS when the process crashed.

    In single-process serial execution, any IN_PROGRESS task at startup is
    unconditionally orphaned — the process that was running it is gone.
    No heartbeat or timing check is needed.
    """

    def __init__(self, store: SQLiteTaskStore) -> None:
        self.store = store

    def recover(self, project_id: str) -> int:
        """
        Reset every task still marked IN_PROGRESS for this project.
          - retries left  → reset to PENDING
          - no retries    → mark FAILED
        Returns number of tasks processed.
        """
        with self.store._conn() as conn:
            crashed = conn.execute(
                """SELECT id, title, retry_count, max_retries
                   FROM tasks
                   WHERE project_id = ? AND status = 'in_progress'""",
                (project_id,)
            ).fetchall()

            for task in crashed:
                if task["retry_count"] < task["max_retries"]:
                    conn.execute(
                        """UPDATE tasks
                           SET status      = 'pending',
                               started_at  = NULL,
                               retry_count = retry_count + 1,
                               error       = 'CRASH: process died mid-execution'
                           WHERE id = ?""",
                        (task["id"],)
                    )
                    console.print(
                        f"[yellow]🔄 Recovered:[/yellow] {task['id']} ({task['title']}) "
                        f"→ PENDING (retry {task['retry_count'] + 1}/{task['max_retries']})"
                    )
                else:
                    conn.execute(
                        """UPDATE tasks
                           SET status = 'failed',
                               error  = 'CRASH: max retries exceeded after repeated crashes'
                           WHERE id = ?""",
                        (task["id"],)
                    )
                    console.print(f"[red]❌ Max retries exhausted:[/red] {task['id']} → FAILED")

        count = len(crashed)
        if count:
            console.print(f"[dim]Recovery complete: {count} task(s) processed.[/dim]\n")
        return count