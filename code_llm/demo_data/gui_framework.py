
"""Simple GUI Framework - Custom implementation for demo."""

class Component:
    """Base class for all GUI components."""

    def __init__(self, id: str):
        self.id = id
        self.children = []
        self.styles = {}

    def add_child(self, child):
        self.children.append(child)
        return self

    def style(self, **kwargs):
        self.styles.update(kwargs)
        return self

    def render(self) -> str:
        raise NotImplementedError


class Button(Component):
    """Clickable button component."""

    def __init__(self, id: str, label: str, on_click: str = ""):
        super().__init__(id)
        self.label = label
        self.on_click = on_click

    def render(self) -> str:
        style_str = "; ".join(f"{k}: {v}" for k, v in self.styles.items())
        return f'<button id="{self.id}" onclick="{self.on_click}" style="{style_str}">{self.label}</button>'


class Display(Component):
    """Text display component."""

    def __init__(self, id: str, value: str = ""):
        super().__init__(id)
        self.value = value

    def render(self) -> str:
        style_str = "; ".join(f"{k}: {v}" for k, v in self.styles.items())
        return f'<div id="{self.id}" class="display" style="{style_str}">{self.value}</div>'


class Container(Component):
    """Container for grouping components."""

    def __init__(self, id: str, layout: str = "vertical"):
        super().__init__(id)
        self.layout = layout

    def render(self) -> str:
        flex_dir = "column" if self.layout == "vertical" else "row"
        style_str = f"display: flex; flex-direction: {flex_dir}; " + "; ".join(f"{k}: {v}" for k, v in self.styles.items())
        children_html = "\n".join(c.render() for c in self.children)
        return f'<div id="{self.id}" style="{style_str}">\n{children_html}\n</div>'


class Input(Component):
    """Text input component."""

    def __init__(self, id: str, placeholder: str = "", on_change: str = ""):
        super().__init__(id)
        self.placeholder = placeholder
        self.on_change = on_change

    def render(self) -> str:
        style_str = "; ".join(f"{k}: {v}" for k, v in self.styles.items())
        return f'<input id="{self.id}" placeholder="{self.placeholder}" onchange="{self.on_change}" style="{style_str}"/>'


class App:
    """Main application container."""

    def __init__(self, title: str = "App"):
        self.title = title
        self.root = Container("root", "vertical")
        self.state = {}
        self.scripts = []

    def add(self, component: Component):
        self.root.add_child(component)
        return self

    def add_script(self, script: str):
        self.scripts.append(script)
        return self

    def render(self) -> str:
        scripts_html = "\n".join(f"<script>{s}</script>" for s in self.scripts)
        return f"""<!DOCTYPE html>
<html>
<head>
    <title>{self.title}</title>
    <style>
        .display {{ font-size: 24px; padding: 10px; background: #f0f0f0; margin: 5px; }}
        button {{ padding: 10px 20px; margin: 5px; font-size: 18px; cursor: pointer; }}
        input {{ padding: 10px; margin: 5px; font-size: 18px; }}
    </style>
</head>
<body>
{self.root.render()}
{scripts_html}
</body>
</html>"""


# Example usage showing how to build a simple counter:
#
# from gui_framework import App, Button, Display, Container
#
# app = App("Counter")
# display = Display("counter-display", "0")
#
# app.add(display)
# app.add(Button("increment", "+", "increment()"))
# app.add(Button("decrement", "-", "decrement()"))
#
# app.add_script("""
# let count = 0;
# function increment() { count++; document.getElementById('counter-display').innerText = count; }
# function decrement() { count--; document.getElementById('counter-display').innerText = count; }
# """)
#
# print(app.render())
