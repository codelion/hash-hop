"""Sign Up Page for DocuSign-like application.

This is the page where users create their account.
MagicLabs Use Case 2: Add password strength meter to this page.
"""

from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
import re

from ..components import FormField, FormFieldProps, Button, ButtonProps, Card, CardProps, Alert, AlertProps
from ..auth import AuthService


@dataclass
class SignUpFormState:
    """State for sign up form."""
    name: str = ""
    email: str = ""
    password: str = ""
    confirm_password: str = ""
    errors: Dict[str, str] = None
    is_submitting: bool = False
    success_message: Optional[str] = None
    error_message: Optional[str] = None

    def __post_init__(self):
        if self.errors is None:
            self.errors = {}


class SignUpPage:
    """Sign up page component."""

    def __init__(self, auth_service: AuthService):
        self.auth_service = auth_service
        self.state = SignUpFormState()

    def validate_email(self, email: str) -> Optional[str]:
        """Validate email format."""
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        if not email:
            return "Email is required"
        if not re.match(pattern, email):
            return "Invalid email format"
        return None

    def validate_password(self, password: str) -> Optional[str]:
        """Validate password requirements."""
        if not password:
            return "Password is required"
        if len(password) < 8:
            return "Password must be at least 8 characters"
        return None

    def validate_form(self) -> bool:
        """Validate the entire form."""
        errors = {}

        if not self.state.name.strip():
            errors["name"] = "Name is required"

        email_error = self.validate_email(self.state.email)
        if email_error:
            errors["email"] = email_error

        password_error = self.validate_password(self.state.password)
        if password_error:
            errors["password"] = password_error

        if self.state.password != self.state.confirm_password:
            errors["confirm_password"] = "Passwords do not match"

        self.state.errors = errors
        return len(errors) == 0

    def handle_submit(self) -> Tuple[bool, str]:
        """Handle form submission."""
        self.state.is_submitting = True
        self.state.success_message = None
        self.state.error_message = None

        if not self.validate_form():
            self.state.is_submitting = False
            return False, "Please fix the errors above"

        success, result = self.auth_service.register(
            email=self.state.email,
            password=self.state.password,
            name=self.state.name
        )

        self.state.is_submitting = False

        if success:
            self.state.success_message = "Account created successfully! Please check your email to verify."
            # Reset form
            self.state.name = ""
            self.state.email = ""
            self.state.password = ""
            self.state.confirm_password = ""
            return True, result
        else:
            self.state.error_message = result
            return False, result

    def render(self) -> Dict[str, Any]:
        """Render the sign up page."""
        # Build form fields
        name_field = FormField(FormFieldProps(
            name="name",
            label="Full Name",
            placeholder="Enter your full name",
            required=True,
            value=self.state.name,
            error=self.state.errors.get("name")
        ))

        email_field = FormField(FormFieldProps(
            name="email",
            label="Email Address",
            type="email",
            placeholder="Enter your email",
            required=True,
            value=self.state.email,
            error=self.state.errors.get("email")
        ))

        password_field = FormField(FormFieldProps(
            name="password",
            label="Password",
            type="password",
            placeholder="Create a password",
            required=True,
            value=self.state.password,
            error=self.state.errors.get("password")
        ))

        confirm_password_field = FormField(FormFieldProps(
            name="confirm_password",
            label="Confirm Password",
            type="password",
            placeholder="Confirm your password",
            required=True,
            value=self.state.confirm_password,
            error=self.state.errors.get("confirm_password")
        ))

        submit_button = Button(ButtonProps(
            text="Create Account",
            variant="primary",
            loading=self.state.is_submitting
        ))

        # Build alerts
        alerts = []
        if self.state.success_message:
            alerts.append(Alert(AlertProps(
                message=self.state.success_message,
                variant="success"
            )).render())
        if self.state.error_message:
            alerts.append(Alert(AlertProps(
                message=self.state.error_message,
                variant="error"
            )).render())

        # Build card
        card = Card(CardProps(
            title="Create Your Account",
            children=[
                *alerts,
                name_field.render(),
                email_field.render(),
                password_field.render(),
                # PASSWORD STRENGTH METER SHOULD BE ADDED HERE
                confirm_password_field.render()
            ],
            footer=submit_button.render()
        ))

        return {
            "type": "div",
            "className": "signup-page",
            "children": [
                {
                    "type": "div",
                    "className": "container",
                    "children": card.render()
                }
            ]
        }
