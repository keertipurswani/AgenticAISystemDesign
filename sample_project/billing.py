"""
billing.py — Plan-based feature gating and usage enforcement.
Each user is on a plan tier that caps how many tasks they can create.
"""

from enum import Enum
from models import User
from database import Database


class PlanTier(Enum):
    FREE = "free"
    PRO = "pro"
    ENTERPRISE = "enterprise"


# Task creation limits per plan_tier
TASK_LIMITS: dict[str, int] = {
    PlanTier.FREE.value: 10,
    PlanTier.PRO.value: 200,
    PlanTier.ENTERPRISE.value: 10_000,
}

# In-memory plan store: user_id → plan_tier
_user_plans: dict[int, str] = {}


class BillingError(Exception):
    pass


class BillingService:
    """Enforces plan_tier limits before allowing resource-creating operations."""

    def __init__(self, db: Database):
        self.db = db

    def assign_plan(self, user: User, plan: PlanTier) -> None:
        _user_plans[user.id] = plan.value

    def get_plan(self, user: User) -> str:
        return _user_plans.get(user.id, PlanTier.FREE.value)

    def check_task_limit(self, user: User) -> None:
        """Raise BillingError if user has reached their plan_tier task cap."""
        plan = self.get_plan(user)
        limit = TASK_LIMITS.get(plan, TASK_LIMITS[PlanTier.FREE.value])
        current_count = len(self.db.list_tasks(owner_id=user.id))
        if current_count >= limit:
            raise BillingError(
                f"Task limit reached for plan '{plan}': {current_count}/{limit}. "
                "Upgrade your plan_tier to create more tasks."
            )

    def usage_summary(self, user: User) -> dict:
        plan = self.get_plan(user)
        limit = TASK_LIMITS.get(plan, TASK_LIMITS[PlanTier.FREE.value])
        used = len(self.db.list_tasks(owner_id=user.id))
        return {"plan_tier": plan, "tasks_used": used, "tasks_limit": limit}
