"""
main.py — Entry point / CLI for the task management app.
Wires all layers together: Database → AuthService → TaskService + supporting services.

Run:
    python main.py
"""

from database import Database
from auth import AuthService
from task_service import TaskService
from permissions import PermissionPolicy, Role
from billing import BillingService, PlanTier
from audit_log import AuditLogger
from notifications import NotificationService
from reports import ReportService
from scheduler import TaskScheduler, RecurrenceInterval
from integrations.email_client import EmailClient
from integrations.slack_webhook import SlackWebhook
from models import Priority


def main():
    # --- wire up all layers ---
    db = Database()
    auth = AuthService(db)
    policy = PermissionPolicy()
    billing = BillingService(db)
    audit = AuditLogger()
    email = EmailClient()
    slack = SlackWebhook()
    notifications = NotificationService(email, slack)
    tasks = TaskService(db, auth, policy, billing, audit, notifications)
    reports = ReportService(db, policy)
    scheduler = TaskScheduler(db, billing, audit)

    # --- seed users ---
    alice = auth.register("alice", "alice@example.com", "pass123")
    bob = auth.register("bob", "bob@example.com", "pass456")
    admin = auth.register("admin", "admin@taskapp.dev", "adminpass")

    policy.assign_role(admin, Role.ADMIN)
    policy.assign_role(alice, Role.MEMBER)
    policy.assign_role(bob, Role.MEMBER)

    billing.assign_plan(alice, PlanTier.PRO)
    billing.assign_plan(bob, PlanTier.FREE)

    alice_token = auth.login("alice@example.com", "pass123")
    bob_token = auth.login("bob@example.com", "pass456")
    admin_token = auth.login("admin@taskapp.dev", "adminpass")

    # --- seed tasks ---
    tasks.create_task(alice_token, "Set up CI pipeline", "Configure GitHub Actions for the repo", Priority.HIGH, tags=["devops"])
    tasks.create_task(alice_token, "Write unit tests", "Cover auth and task_service modules", Priority.MEDIUM, tags=["testing"])
    tasks.create_task(bob_token, "Design database schema", "Plan the production PostgreSQL schema", Priority.HIGH, tags=["database"])

    # --- complete a task ---
    alice_tasks = tasks.list_my_tasks(alice_token)
    tasks.complete_task(alice_token, alice_tasks[0].id)

    # --- set up a recurring scheduled task for alice ---
    scheduler.add_schedule(
        owner=alice,
        title="Weekly status report",
        description="Summarise progress and blockers for the team",
        priority=Priority.LOW,
        interval=RecurrenceInterval.WEEKLY,
        tags=["recurring"],
    )

    # --- print summary ---
    print("=== Alice's tasks ===")
    for t in tasks.list_my_tasks(alice_token):
        print(f"  [{t.status.value}] {t.title} (priority={t.priority.value})")

    print("\n=== Bob's tasks ===")
    for t in tasks.list_my_tasks(bob_token):
        print(f"  [{t.status.value}] {t.title} (priority={t.priority.value})")

    print("\n=== Search 'test' ===")
    for t in tasks.search_tasks(alice_token, "test"):
        print(f"  {t.title}")

    print("\n=== System report (admin) ===")
    r = reports.system_report(admin)
    print(f"  Users: {r.total_users}, Tasks: {r.total_tasks}, Done: {r.completed_tasks}")
    print(f"  By priority: {r.tasks_by_priority}")

    print("\n=== Audit log ===")
    for entry in audit.all_entries():
        print(f"  [{entry.event_type.value}] actor={entry.actor_id} target={entry.target_id}")


if __name__ == "__main__":
    main()
