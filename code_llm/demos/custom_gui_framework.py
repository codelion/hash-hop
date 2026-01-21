"""CustomGUI: A simple in-context GUI framework.

This is a NOVEL framework that doesn't exist anywhere else.
The model must learn it purely from context to build applications.

This demonstrates MagicLabs Use Case 1: In-context learning with
a custom framework the model has never seen before.
"""

from typing import Callable, Dict, List, Optional, Any
from dataclasses import dataclass, field


@dataclass
class Style:
    """Style properties for widgets."""
    width: int = 100
    height: int = 30
    bg_color: str = "#ffffff"
    fg_color: str = "#000000"
    font_size: int = 14
    padding: int = 5
    border: bool = False
    border_color: str = "#cccccc"


@dataclass
class Widget:
    """Base widget class."""
    id: str
    style: Style = field(default_factory=Style)
    visible: bool = True
    enabled: bool = True

    def render(self) -> str:
        """Render widget to string representation."""
        raise NotImplementedError

    def handle_event(self, event: str, data: Any = None) -> None:
        """Handle user interaction event."""
        pass


@dataclass
class Label(Widget):
    """Text label widget."""
    text: str = ""

    def render(self) -> str:
        if not self.visible:
            return ""
        return f"[Label:{self.id}] {self.text}"


@dataclass
class Button(Widget):
    """Clickable button widget."""
    text: str = "Button"
    on_click: Optional[Callable] = None

    def render(self) -> str:
        if not self.visible:
            return ""
        state = "enabled" if self.enabled else "disabled"
        return f"[Button:{self.id}:{state}] {self.text}"

    def handle_event(self, event: str, data: Any = None) -> None:
        if event == "click" and self.enabled and self.on_click:
            self.on_click()


@dataclass
class TextInput(Widget):
    """Text input field widget."""
    value: str = ""
    placeholder: str = ""
    on_change: Optional[Callable[[str], None]] = None

    def render(self) -> str:
        if not self.visible:
            return ""
        display = self.value if self.value else f"({self.placeholder})"
        return f"[Input:{self.id}] {display}"

    def handle_event(self, event: str, data: Any = None) -> None:
        if event == "input" and self.enabled:
            self.value = str(data) if data else ""
            if self.on_change:
                self.on_change(self.value)


@dataclass
class Container(Widget):
    """Container that holds other widgets."""
    children: List[Widget] = field(default_factory=list)
    layout: str = "vertical"  # "vertical" or "horizontal"

    def add(self, widget: Widget) -> None:
        """Add a child widget."""
        self.children.append(widget)

    def remove(self, widget_id: str) -> None:
        """Remove a child widget by id."""
        self.children = [w for w in self.children if w.id != widget_id]

    def get(self, widget_id: str) -> Optional[Widget]:
        """Get a child widget by id."""
        for child in self.children:
            if child.id == widget_id:
                return child
            if isinstance(child, Container):
                found = child.get(widget_id)
                if found:
                    return found
        return None

    def render(self) -> str:
        if not self.visible:
            return ""
        separator = "\n" if self.layout == "vertical" else " | "
        rendered = [c.render() for c in self.children if c.visible]
        return separator.join(rendered)


class App:
    """Main application class."""

    def __init__(self, title: str = "CustomGUI App"):
        self.title = title
        self.root = Container(id="root", layout="vertical")
        self.state: Dict[str, Any] = {}
        self._running = False

    def set_state(self, key: str, value: Any) -> None:
        """Set application state."""
        self.state[key] = value

    def get_state(self, key: str, default: Any = None) -> Any:
        """Get application state."""
        return self.state.get(key, default)

    def add_widget(self, widget: Widget) -> None:
        """Add widget to root container."""
        self.root.add(widget)

    def get_widget(self, widget_id: str) -> Optional[Widget]:
        """Get widget by id."""
        return self.root.get(widget_id)

    def dispatch_event(self, widget_id: str, event: str, data: Any = None) -> None:
        """Dispatch event to a widget."""
        widget = self.get_widget(widget_id)
        if widget:
            widget.handle_event(event, data)

    def render(self) -> str:
        """Render the entire application."""
        header = f"=== {self.title} ==="
        body = self.root.render()
        return f"{header}\n{body}\n{'=' * len(header)}"

    def run(self) -> None:
        """Run the application (prints current state)."""
        self._running = True
        print(self.render())


# Example: Simple counter app
def create_counter_app() -> App:
    """Example: Create a simple counter application."""
    app = App(title="Counter")
    app.set_state("count", 0)

    label = Label(id="count_label", text="Count: 0")

    def increment():
        count = app.get_state("count", 0) + 1
        app.set_state("count", count)
        label.text = f"Count: {count}"

    def decrement():
        count = app.get_state("count", 0) - 1
        app.set_state("count", count)
        label.text = f"Count: {count}"

    app.add_widget(label)
    app.add_widget(Button(id="inc_btn", text="+", on_click=increment))
    app.add_widget(Button(id="dec_btn", text="-", on_click=decrement))

    return app


# Example: Simple form app
def create_form_app() -> App:
    """Example: Create a simple form application."""
    app = App(title="User Form")

    name_input = TextInput(id="name", placeholder="Enter name")
    email_input = TextInput(id="email", placeholder="Enter email")
    result_label = Label(id="result", text="")

    def submit():
        name = name_input.value
        email = email_input.value
        result_label.text = f"Submitted: {name} ({email})"

    app.add_widget(Label(id="name_label", text="Name:"))
    app.add_widget(name_input)
    app.add_widget(Label(id="email_label", text="Email:"))
    app.add_widget(email_input)
    app.add_widget(Button(id="submit_btn", text="Submit", on_click=submit))
    app.add_widget(result_label)

    return app
