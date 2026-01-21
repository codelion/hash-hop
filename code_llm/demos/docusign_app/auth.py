"""Authentication module for DocuSign-like application.

This simulates a real-world codebase for MagicLabs Use Case 2:
Adding a password strength meter to an existing application.
"""

import hashlib
import secrets
import re
from typing import Optional, Dict, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta


@dataclass
class User:
    """User model."""
    id: str
    email: str
    password_hash: str
    name: str
    created_at: datetime
    last_login: Optional[datetime] = None
    is_verified: bool = False
    is_active: bool = True


class PasswordHasher:
    """Handles password hashing and verification."""

    def __init__(self, salt_length: int = 32):
        self.salt_length = salt_length

    def hash_password(self, password: str) -> str:
        """Hash a password with a random salt."""
        salt = secrets.token_hex(self.salt_length)
        hash_obj = hashlib.pbkdf2_hmac(
            'sha256',
            password.encode('utf-8'),
            salt.encode('utf-8'),
            100000
        )
        return f"{salt}${hash_obj.hex()}"

    def verify_password(self, password: str, stored_hash: str) -> bool:
        """Verify a password against stored hash."""
        try:
            salt, hash_value = stored_hash.split('$')
            hash_obj = hashlib.pbkdf2_hmac(
                'sha256',
                password.encode('utf-8'),
                salt.encode('utf-8'),
                100000
            )
            return hash_obj.hex() == hash_value
        except ValueError:
            return False


class AuthService:
    """Authentication service."""

    def __init__(self):
        self.hasher = PasswordHasher()
        self.users: Dict[str, User] = {}
        self.sessions: Dict[str, str] = {}  # token -> user_id

    def register(self, email: str, password: str, name: str) -> Tuple[bool, str]:
        """Register a new user."""
        # Check if email already exists
        for user in self.users.values():
            if user.email == email:
                return False, "Email already registered"

        # Create user
        user_id = secrets.token_hex(16)
        user = User(
            id=user_id,
            email=email,
            password_hash=self.hasher.hash_password(password),
            name=name,
            created_at=datetime.now()
        )
        self.users[user_id] = user
        return True, user_id

    def login(self, email: str, password: str) -> Tuple[bool, str]:
        """Login a user and return session token."""
        # Find user by email
        user = None
        for u in self.users.values():
            if u.email == email:
                user = u
                break

        if not user:
            return False, "Invalid email or password"

        if not user.is_active:
            return False, "Account is disabled"

        if not self.hasher.verify_password(password, user.password_hash):
            return False, "Invalid email or password"

        # Create session
        token = secrets.token_hex(32)
        self.sessions[token] = user.id
        user.last_login = datetime.now()

        return True, token

    def logout(self, token: str) -> bool:
        """Logout a user by invalidating their session."""
        if token in self.sessions:
            del self.sessions[token]
            return True
        return False

    def get_current_user(self, token: str) -> Optional[User]:
        """Get current user from session token."""
        user_id = self.sessions.get(token)
        if user_id:
            return self.users.get(user_id)
        return None

    def change_password(self, user_id: str, old_password: str, new_password: str) -> Tuple[bool, str]:
        """Change user password."""
        user = self.users.get(user_id)
        if not user:
            return False, "User not found"

        if not self.hasher.verify_password(old_password, user.password_hash):
            return False, "Current password is incorrect"

        user.password_hash = self.hasher.hash_password(new_password)
        return True, "Password changed successfully"
