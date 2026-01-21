"""UI Components for DocuSign-like application.

React-style components for the document signing app.
"""

from typing import Optional, List, Dict, Any, Callable
from dataclasses import dataclass, field


@dataclass
class FormFieldProps:
    """Props for form field component."""
    name: str
    label: str
    type: str = "text"
    placeholder: str = ""
    required: bool = False
    disabled: bool = False
    value: str = ""
    error: Optional[str] = None
    on_change: Optional[Callable[[str], None]] = None


class FormField:
    """Form field component."""

    def __init__(self, props: FormFieldProps):
        self.props = props

    def render(self) -> Dict[str, Any]:
        """Render the form field."""
        return {
            "type": "div",
            "className": "form-field",
            "children": [
                {
                    "type": "label",
                    "htmlFor": self.props.name,
                    "children": self.props.label
                },
                {
                    "type": "input",
                    "id": self.props.name,
                    "name": self.props.name,
                    "type": self.props.type,
                    "placeholder": self.props.placeholder,
                    "required": self.props.required,
                    "disabled": self.props.disabled,
                    "value": self.props.value,
                    "className": "form-input" + (" error" if self.props.error else "")
                },
                {
                    "type": "span",
                    "className": "error-message",
                    "children": self.props.error or ""
                } if self.props.error else None
            ]
        }


@dataclass
class ButtonProps:
    """Props for button component."""
    text: str
    variant: str = "primary"  # primary, secondary, danger
    disabled: bool = False
    loading: bool = False
    on_click: Optional[Callable] = None


class Button:
    """Button component."""

    def __init__(self, props: ButtonProps):
        self.props = props

    def render(self) -> Dict[str, Any]:
        """Render the button."""
        class_name = f"btn btn-{self.props.variant}"
        if self.props.disabled:
            class_name += " disabled"
        if self.props.loading:
            class_name += " loading"

        return {
            "type": "button",
            "className": class_name,
            "disabled": self.props.disabled or self.props.loading,
            "children": "Loading..." if self.props.loading else self.props.text
        }


@dataclass
class AlertProps:
    """Props for alert component."""
    message: str
    variant: str = "info"  # info, success, warning, error
    dismissible: bool = True
    on_dismiss: Optional[Callable] = None


class Alert:
    """Alert component."""

    def __init__(self, props: AlertProps):
        self.props = props

    def render(self) -> Dict[str, Any]:
        """Render the alert."""
        return {
            "type": "div",
            "className": f"alert alert-{self.props.variant}",
            "role": "alert",
            "children": [
                {"type": "span", "children": self.props.message},
                {
                    "type": "button",
                    "className": "alert-dismiss",
                    "children": "×"
                } if self.props.dismissible else None
            ]
        }


@dataclass
class CardProps:
    """Props for card component."""
    title: str
    children: List[Any] = field(default_factory=list)
    footer: Optional[Any] = None


class Card:
    """Card component."""

    def __init__(self, props: CardProps):
        self.props = props

    def render(self) -> Dict[str, Any]:
        """Render the card."""
        return {
            "type": "div",
            "className": "card",
            "children": [
                {
                    "type": "div",
                    "className": "card-header",
                    "children": {"type": "h3", "children": self.props.title}
                },
                {
                    "type": "div",
                    "className": "card-body",
                    "children": self.props.children
                },
                {
                    "type": "div",
                    "className": "card-footer",
                    "children": self.props.footer
                } if self.props.footer else None
            ]
        }
