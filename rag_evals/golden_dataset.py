"""
golden_dataset.py — Ground-truth evaluation set for the sample_project codebase.

Questions reflect what a real engineer would ask when onboarding to or auditing
this codebase. Reference answers are verified against the actual source files.

Used by all RAG eval files so results are directly comparable across strategies.
"""

TEST_CASES = [
    {
        "question": "What is AuthService responsible for?",
        "reference": (
            "AuthService handles user registration, login, session management, "
            "and basic access control. It stores sessions in an in-memory dictionary "
            "mapping tokens to user IDs, validates credentials on login, and resolves "
            "session tokens to User objects via get_current_user."
        ),
    },
    {
        "question": "What protects against privilege escalation in this codebase?",
        "reference": (
            "PermissionPolicy in permissions.py enforces role-based access control. "
            "ROLE_PERMISSIONS maps each role (admin, member, viewer) to a set of allowed "
            "actions. PermissionPolicy.require(user, action) raises PermissionError if the "
            "user's role does not include the requested action. Only ADMIN can perform "
            "task:assign; VIEWER is restricted to task:view only."
        ),
    },
    {
        "question": "What happens if a user provides an incorrect password during login?",
        "reference": (
            "AuthService.login in auth.py compares the submitted password against the "
            "stored plaintext password in the _passwords dict. If they do not match, "
            "it raises AuthError('Invalid password')."
        ),
    },
    {
        "question": "How does the system prevent a user from creating unlimited tasks?",
        "reference": (
            "BillingService.check_task_limit in billing.py looks up the user's plan tier "
            "from _user_plans and the corresponding cap from TASK_LIMITS "
            "(FREE=10, PRO=200, ENTERPRISE=10000). It counts current tasks via "
            "db.list_tasks(owner_id=user.id) and raises BillingError if the count "
            "is at or above the limit."
        ),
    },
    {
        "question": "Find every place BillingError is raised in the codebase.",
        "reference": (
            "BillingError is raised in exactly one place: BillingService.check_task_limit "
            "in billing.py, when the user's current task count is greater than or equal "
            "to their plan limit."
        ),
    },
    {
        "question": "Where is _user_plans defined and what does it store?",
        "reference": (
            "_user_plans is a module-level dictionary defined in billing.py. "
            "It maps user_id (int) to plan_tier (str) and serves as the in-memory "
            "plan store. It is written by BillingService.assign_plan and read by "
            "BillingService.get_plan."
        ),
    },
    {
        "question": "Which file imports both Task and Status?",
        "reference": (
            "task_service.py imports both Task and Status: "
            "'from models import Task, Priority, Status'."
        ),
    },
    {
        "question": "Find every place AuditEventType.PERMISSION_DENIED is recorded.",
        "reference": (
            "AuditEventType.PERMISSION_DENIED is defined in audit_log.py but is not "
            "explicitly recorded anywhere in the current codebase. The enum value exists "
            "for future use; no call to audit.record(AuditEventType.PERMISSION_DENIED, ...) "
            "appears in task_service.py or any other file."
        ),
    },
    {
        "question": "Which methods in the codebase check is_active before proceeding?",
        "reference": (
            "AuthService.get_current_user in auth.py checks 'if not user or not user.is_active' "
            "after resolving the session token, raising AuthError('User account is inactive') "
            "if the flag is False. No other method checks is_active directly."
        ),
    },
    {
        "question": "What are all the methods inside AuthService?",
        "reference": (
            "AuthService in auth.py has five methods: "
            "__init__ (initialises db and _sessions), "
            "register (creates a new user), "
            "login (validates credentials and returns a session token), "
            "get_current_user (resolves a token to a User object), "
            "logout (invalidates a session token)."
        ),
    },
    {
        "question": "What are all the methods inside TaskService?",
        "reference": (
            "TaskService in task_service.py has six methods: "
            "create_task, complete_task, list_my_tasks, delete_task, assign_task, search_tasks."
        ),
    },
    {
        "question": "What fields does the Task dataclass have?",
        "reference": (
            "Task in models.py has ten fields: id (int), title (str), description (str), "
            "owner_id (int), priority (Priority, default MEDIUM), status (Status, default TODO), "
            "due_date (Optional[datetime], default None), created_at (datetime), "
            "updated_at (datetime), tags (list[str], default empty list). "
            "It also has an is_overdue() method."
        ),
    },
    {
        "question": (
            "Trace the complete call chain when create_task is called — "
            "list every method invoked across all files in order."
        ),
        "reference": (
            "1. TaskService.create_task (task_service.py) is the entry point. "
            "2. AuthService.get_current_user (auth.py) — resolves token to User. "
            "3. PermissionPolicy.require (permissions.py) — checks 'task:create' permission. "
            "4. BillingService.check_task_limit (billing.py) — enforces plan cap. "
            "5. Database.create_task (database.py) — persists the task. "
            "6. AuditLogger.record (audit_log.py) — writes TASK_CREATED event. "
            "7. NotificationService.on_task_created (notifications.py) — sends email + Slack."
        ),
    },
    {
        "question": (
            "If complete_task is called with an expired session token, "
            "trace the full call chain and identify exactly where and how it fails."
        ),
        "reference": (
            "1. TaskService.complete_task (task_service.py) calls self.auth.get_current_user(token). "
            "2. AuthService.get_current_user (auth.py) calls self._sessions.get(token). "
            "3. If the token is not in _sessions, user_id is None. "
            "4. The check 'if user_id is None' triggers and raises "
            "AuthError('Invalid or expired session token'). "
            "Execution stops here — policy, billing, db, audit, and notifications are never reached."
        ),
    },
    {
        "question": (
            "If a user's account is deactivated (is_active = False), "
            "which specific service methods would they be blocked from calling?"
        ),
        "reference": (
            "Every method that calls auth.get_current_user(token) will block a deactivated user, "
            "because get_current_user raises AuthError('User account is inactive') when user.is_active is False. "
            "The blocked methods in TaskService are: create_task, complete_task, list_my_tasks, "
            "delete_task, assign_task, and search_tasks — all six, since every one of them "
            "starts by resolving the session token."
        ),
    },
    {
        "question": "Which TaskService methods check that the requesting user owns the task before proceeding?",
        "reference": (
            "complete_task and delete_task in task_service.py both check ownership. "
            "After retrieving the task, each method compares task.owner_id != user.id "
            "and raises AuthError if the requesting user is not the owner. "
            "assign_task does NOT check prior ownership — it requires task:assign permission "
            "(admin only) and changes the owner_id to the assignee."
        ),
    },
    {
        "question": "What audit events are recorded across the entire task lifecycle?",
        "reference": (
            "Four audit events are recorded for tasks: "
            "AuditEventType.TASK_CREATED (in create_task), "
            "AuditEventType.TASK_COMPLETED (in complete_task), "
            "AuditEventType.TASK_DELETED (in delete_task, only if deletion succeeded), "
            "AuditEventType.TASK_ASSIGNED (in assign_task). "
            "All are recorded via AuditLogger.record in audit_log.py."
        ),
    },
    {
        "question": "What notifications are sent when a task is assigned to a new user?",
        "reference": (
            "NotificationService.on_task_assigned in notifications.py sends: "
            "1. An email to the assignee's address with subject 'Task assigned to you: <title>' "
            "and body naming the assigner. "
            "2. A Slack message: '<assigner> assigned <title> to <assignee>'. "
            "No notification is sent to the original owner."
        ),
    },
]
