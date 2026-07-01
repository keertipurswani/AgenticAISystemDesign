from __future__ import annotations

from rich.console import Console
from rich.table import Table

from educosys_claude.config import config
from educosys_claude.tasks.task_store import SQLiteTaskStore

console = Console()

_STATUS_STYLE = {
    "completed":   "green",
    "failed":      "red",
    "in_progress": "yellow",
    "pending":     "dim",
    "blocked":     "red",
    "skipped":     "dim",
}


def show_task_status() -> None:
    """Print a Rich table of all tasks for the latest approved project."""
    db_path    = config.get("tasks", {}).get("db_path", ".educosys/tasks.db")
    store      = SQLiteTaskStore(db_path)
    project_id = store.get_latest_approved_project()

    if not project_id:
        console.print("[yellow]No active project found. Run /plan <goal> first.[/yellow]")
        return

    tasks    = store.get_all_tasks(project_id)
    progress = store.get_progress(project_id)
    total    = sum(progress.values())
    done     = progress.get("completed", 0)

    console.print(f"\n[bold]Project:[/bold] {project_id}")
    console.print(f"[dim]Progress: {done}/{total} completed[/dim]\n")

    table = Table(show_header=True, header_style="bold")
    table.add_column("#",       width=4)
    table.add_column("ID",      width=12)
    table.add_column("Type",    width=10)
    table.add_column("Title",   width=35)
    table.add_column("Status",  width=12)
    table.add_column("Retries", width=8)
    table.add_column("Error",   width=40)

    for i, task in enumerate(tasks, 1):
        style  = _STATUS_STYLE.get(task["status"], "")
        error  = (task.get("error") or "")[:60]
        table.add_row(
            str(i),
            task["id"],
            task["task_type"],
            task["title"],
            f"[{style}]{task['status']}[/{style}]",
            f"{task['retry_count']}/{task['max_retries']}",
            error,
        )

    console.print(table)