"""
permissions.py — Role-based access control policy.
Defines what each role (admin, member, viewer) can do.
"""

from enum import Enum
from models import User


class Role(Enum):
    ADMIN = "admin"
    MEMBER = "member"
    VIEWER = "viewer"


# Maps role → set of allowed action strings
ROLE_PERMISSIONS: dict[str, set[str]] = {
    Role.ADMIN.value: {
        "task:create", "task:complete", "task:delete", "task:view",
        "task:assign", "report:view", "user:manage", "billing:view",
    },
    Role.MEMBER.value: {
        "task:create", "task:complete", "task:delete", "task:view",
        "report:view",
    },
    Role.VIEWER.value: {
        "task:view",
    },
}

# In-memory role store: user_id → role (default MEMBER)
_user_roles: dict[int, str] = {}


class PermissionError(Exception):
    pass


class PermissionPolicy:
    """Checks whether a user is allowed to perform a given action."""

    def assign_role(self, user: User, role: Role) -> None:
        _user_roles[user.id] = role.value

    def get_role(self, user: User) -> str:
        return _user_roles.get(user.id, Role.MEMBER.value)

    def require(self, user: User, action: str) -> None:
        """Raise PermissionError if user's role does not allow action."""
        role = self.get_role(user)
        allowed = ROLE_PERMISSIONS.get(role, set())
        if action not in allowed:
            raise PermissionError(
                f"User '{user.username}' with role '{role}' cannot perform '{action}'"
            )

    def can(self, user: User, action: str) -> bool:
        role = self.get_role(user)
        return action in ROLE_PERMISSIONS.get(role, set())
