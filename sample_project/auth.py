"""
auth.py — Simple authentication layer.
Uses plaintext passwords for demo simplicity.
NOTE: In production always hash passwords (e.g. bcrypt).
"""

from database import Database
from models import User


class AuthError(Exception):
    """Raised when authentication or authorisation fails."""
    pass


class AuthService:
    """Handles login and basic access control."""

    def __init__(self, db: Database):
        self.db = db
        # In-memory session store: token → user_id
        # In production use JWT or a proper session backend
        self._sessions: dict[str, int] = {}

    def register(self, username: str, email: str, password: str) -> User:
        """Create a new user. Raises AuthError if email already exists."""
        if self.db.get_user_by_email(email):
            raise AuthError(f"Email already registered: {email}")
        user = self.db.create_user(username, email)
        # Store password against user id (plaintext — demo only!)
        self._passwords = getattr(self, "_passwords", {})
        self._passwords[user.id] = password
        return user

    def login(self, email: str, password: str) -> str:
        """
        Validate credentials and return a session token.
        Token is just the user id as a string — demo only.
        """
        user = self.db.get_user_by_email(email)
        if not user:
            raise AuthError("User not found")
        stored = getattr(self, "_passwords", {}).get(user.id)
        if stored != password:
            raise AuthError("Invalid password")
        token = f"token-{user.id}"
        self._sessions[token] = user.id
        return token

    def get_current_user(self, token: str) -> User:
        """Resolve a session token to a User. Raises AuthError if invalid."""
        user_id = self._sessions.get(token)
        if user_id is None:
            raise AuthError("Invalid or expired session token")
        user = self.db.get_user(user_id)
        if not user or not user.is_active:
            raise AuthError("User account is inactive")
        return user

    def logout(self, token: str) -> None:
        """Invalidate a session token."""
        self._sessions.pop(token, None)
