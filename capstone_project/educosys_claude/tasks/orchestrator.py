from __future__ import annotations

import asyncio
import json

from rich.console import Console

from pathlib import Path

from educosys_claude.config import config
from educosys_claude.tasks.task_store import SQLiteTaskStore
from educosys_claude.tasks.executor import run_subtask_agent
from educosys_claude.tasks.planner import create_plan
from educosys_claude.tasks.approval import present_plan_for_approval
from educosys_claude.tasks.recovery import RecoveryManager
from educosys_claude.context.indexers.factory import get_indexer
from educosys_claude.observability.logger import get_logger

logger = get_logger(__name__)
console = Console()


class TaskOrchestrator:
    """
    Main execution loop: finds tasks whose dependencies are met,
    claims them atomically, and dispatches them to subtask agents.

    Serial by default (max_concurrent=1) for cost control and determinism.
    """

    def __init__(self, store: SQLiteTaskStore, max_concurrent: int = 1) -> None:
        self.store = store
        self.max_concurrent = max_concurrent

    async def run(self, project_id: str) -> None:
        """
        Loop until all tasks reach a terminal state.

        Each iteration:
          1. Check progress — exit if nothing pending or in-progress
          2. Get ready tasks (all deps completed/skipped)
          3. Dispatch up to max_concurrent tasks
          4. Sleep 5s and repeat if waiting on in-progress tasks
        """
        console.print(f"\n[bold blue]Starting execution for project {project_id}[/bold blue]\n")

        while True:
            progress    = self.store.get_progress(project_id)
            pending     = progress.get("pending", 0)
            in_progress = progress.get("in_progress", 0)
            completed   = progress.get("completed", 0)
            failed      = progress.get("failed", 0)
            total       = sum(progress.values())

            console.print(
                f"[dim]Progress: {completed}/{total} completed"
                f" · {in_progress} in-progress"
                f" · {pending} pending"
                f" · {failed} failed[/dim]"
            )
            # all tasks are done - nothing in pending and nothing is in progress
            if pending == 0 and in_progress == 0:
                _print_final_summary(progress)
                break

            ready = self.store.get_ready_tasks(project_id)
            if not ready:
                if in_progress > 0:
                    console.print("[dim]⏳ Waiting for in-progress tasks...[/dim]")
                else:
                    console.print("[yellow]⚠ No tasks are ready — some may be blocked by failed dependencies.[/yellow]")
                    console.print("[yellow]  Use /task_status to inspect.[/yellow]")
                    break
                await asyncio.sleep(5)
                continue

            batch = ready[:self.max_concurrent]
            await asyncio.gather(*[self._execute(task) for task in batch])

    async def _execute(self, task: dict) -> None:
        """Claim and execute a single task, handling retries via fail_task."""
        console.print(f"\n[bold]▶ Starting:[/bold] [{task['id']}] {task['title']}")

        if not self.store.claim_task(task["id"]):
            console.print(f"[dim]⚠ Task {task['id']} already claimed — skipping[/dim]")
            return

        try:
            # Fetch what dependency tasks actually produced and inject into the agent.
            dep_ids     = json.loads(task.get("depends_on") or "[]")
            dep_outputs = self.store.get_dep_results(dep_ids)

            result = await run_subtask_agent(task, dep_outputs=dep_outputs)
            self.store.complete_task(task["id"], result)
            console.print(f"[green]✅ Completed:[/green] [{task['id']}] {task['title']}")

        except Exception as e:
            error_msg = str(e)
            self.store.fail_task(task["id"], error_msg)
            console.print(f"[red]❌ Failed:[/red]    [{task['id']}] {task['title']}: {error_msg[:120]}")
            logger.error(f"Task {task['id']} failed: {error_msg}")


async def handle_plan_command(goal: str) -> None:
    """
    Full /plan flow — entry point called by main.py.

      1. Check DB for an existing approved project → resume + recover if found
      2. Otherwise: plan → human approval loop → persist → execute
    """
    db_path = config.get("tasks", {}).get("db_path", ".educosys/tasks.db")
    store   = SQLiteTaskStore(db_path)
    recover = RecoveryManager(store)

    project_id = store.get_latest_approved_project()
    if project_id:
        console.print(f"\n[yellow]↩ Resuming existing project {project_id}...[/yellow]")
        recovered = recover.recover(project_id)
        if recovered:
            console.print(f"[dim]Recovered {recovered} crashed task(s)[/dim]")
    else:
        console.print("\n[dim]Planning with LLM (this may take a moment)...[/dim]")
        extra_context = ""
        approved_plan = None

        while approved_plan is None:
            raw_plan      = create_plan(goal, extra_context)
            approved_plan = present_plan_for_approval(raw_plan)
            if approved_plan is None:
                extra_context = input("What should change in the re-plan?\n> ").strip()
                console.print("\n[dim]Re-planning with your feedback...[/dim]")

        project_id = store.create_project(goal, approved_plan)
        console.print(f"\n[dim]Project {project_id} saved.[/dim]")

    orchestrator = TaskOrchestrator(store, max_concurrent=1)
    await orchestrator.run(project_id)

    # Re-index the project directory so /ask follow-up questions can find
    # the files that were just generated by the plan tasks.
    console.print("\n[dim]Re-indexing generated files so /ask can query them...[/dim]")
    try:
        get_indexer()(str(Path.cwd()))
        console.print("[green]✓ Index updated — you can now use /ask to ask about the generated code.[/green]")
    except Exception as e:
        logger.warning(f"Re-index after /plan failed: {e}")
        console.print(f"[yellow]⚠ Re-index failed: {e}[/yellow]")


def _print_final_summary(progress: dict[str, int]) -> None:
    completed = progress.get("completed", 0)
    failed    = progress.get("failed", 0)
    blocked   = progress.get("blocked", 0)
    skipped   = progress.get("skipped", 0)

    if failed == 0 and blocked == 0:
        console.print(f"\n[bold green]🎉 All {completed} tasks completed successfully![/bold green]")
    else:
        console.print(f"\n[bold yellow]⚠ Execution finished with issues:[/bold yellow]")
        console.print(f"  ✅ Completed: {completed}")
        if failed:
            console.print(f"  ❌ Failed:    {failed}  (run /task_status to review)")
        if blocked:
            console.print(f"  🚫 Blocked:   {blocked}  (dependencies failed)")
        if skipped:
            console.print(f"  ⏭ Skipped:   {skipped}")