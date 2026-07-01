from __future__ import annotations

from rich.console import Console
from rich.table import Table

from educosys_claude.tasks.planner import ExecutionPlan, PlannedTask

console = Console()


def present_plan_for_approval(plan: ExecutionPlan) -> ExecutionPlan | None:
    """
    Render the plan as a Rich table and prompt the user to:
      [A] Approve — returns the plan as-is
      [M] Modify  — edit a task description in-place and re-render
      [R] Reject  — returns None so the caller re-plans with feedback

    Loop continues until the user approves or rejects.
    """
    while True:
        _render_plan(plan)
        choice = input("\n[A]pprove / [M]odify task / [R]eject and re-plan: ").strip().upper()

        if choice == "A":
            return plan

        elif choice == "M":
            task_id = input("Enter task ID to modify: ").strip()
            task = next((t for t in plan.tasks if t.id == task_id), None)
            if not task:
                console.print(f"[red]Task '{task_id}' not found.[/red]")
                continue
            console.print(f"\nCurrent description:\n{task.description}\n")
            new_desc = input("New description: ").strip()
            if new_desc:
                task.description = new_desc
            console.print("[green]Task updated.[/green]")

        elif choice == "R":
            return None

        else:
            console.print("[yellow]Please enter A, M, or R.[/yellow]")


def _render_plan(plan: ExecutionPlan) -> None:
    console.print(f"\n[bold blue]Plan: {plan.project_name}[/bold blue]")
    console.print(f"[dim]{plan.goal_summary}[/dim]")
    console.print(f"[dim]Stack: {', '.join(plan.tech_stack)} | Est: {plan.total_estimated_hours}h[/dim]\n")

    table = Table(show_header=True, header_style="bold")
    table.add_column("ID",          style="dim",    width=12)
    table.add_column("Type",        width=10)
    table.add_column("Title",       width=32)
    table.add_column("Depends on",  width=20)
    table.add_column("Output files",width=30)

    for task in plan.tasks:
        table.add_row(
            task.id,
            task.task_type.value,
            task.title,
            ", ".join(task.depends_on) or "—",
            "\n".join(task.output_files) or "—",
        )

    console.print(table)

    if plan.risks:
        console.print("\n[yellow]Risks:[/yellow]")
        for r in plan.risks:
            console.print(f"  • {r}")