
"""Calculator built with the custom GUI framework."""

from gui_framework import App, Button, Display, Container

# Create the calculator app
app = App("Calculator")

# Create display
display = Display("calc-display", "0").style(
    font_size="32px",
    text_align="right",
    min_width="200px"
)
app.add(display)

# Create button rows
row1 = Container("row1", "horizontal")
row1.add_child(Button("btn-7", "7", "press('7')"))
row1.add_child(Button("btn-8", "8", "press('8')"))
row1.add_child(Button("btn-9", "9", "press('9')"))
row1.add_child(Button("btn-div", "/", "press('/')"))
app.add(row1)

row2 = Container("row2", "horizontal")
row2.add_child(Button("btn-4", "4", "press('4')"))
row2.add_child(Button("btn-5", "5", "press('5')"))
row2.add_child(Button("btn-6", "6", "press('6')"))
row2.add_child(Button("btn-mul", "*", "press('*')"))
app.add(row2)

row3 = Container("row3", "horizontal")
row3.add_child(Button("btn-1", "1", "press('1')"))
row3.add_child(Button("btn-2", "2", "press('2')"))
row3.add_child(Button("btn-3", "3", "press('3')"))
row3.add_child(Button("btn-sub", "-", "press('-')"))
app.add(row3)

row4 = Container("row4", "horizontal")
row4.add_child(Button("btn-0", "0", "press('0')"))
row4.add_child(Button("btn-clear", "C", "clear()"))
row4.add_child(Button("btn-eq", "=", "calculate()"))
row4.add_child(Button("btn-add", "+", "press('+')"))
app.add(row4)

# Add calculator logic
app.add_script("""
let expression = '';

function press(val) {
    expression += val;
    document.getElementById('calc-display').innerText = expression;
}

function clear() {
    expression = '';
    document.getElementById('calc-display').innerText = '0';
}

function calculate() {
    try {
        let result = eval(expression);
        document.getElementById('calc-display').innerText = result;
        expression = String(result);
    } catch (e) {
        document.getElementById('calc-display').innerText = 'Error';
        expression = '';
    }
}
""")

# Render the app
print(app.render())
