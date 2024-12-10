from flask import Flask, render_template, request, redirect, url_for
import sqlite3

app = Flask(__name__)
DATABASE = 'tasks.db'


def init_db():
    """Initialize the SQLite database."""
    with sqlite3.connect(DATABASE) as conn:
        cursor = conn.cursor()
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS tasks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            description TEXT NOT NULL,
            status INTEGER NOT NULL
        )
        """)
    print("Database initialized.")


@app.route('/')
def home():
    """Display all tasks."""
    with sqlite3.connect(DATABASE) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM tasks")
        tasks = cursor.fetchall()
    return render_template('home.html', tasks=tasks)


@app.route('/create', methods=['GET', 'POST'])
def create():
    """Create a new task."""
    if request.method == 'POST':
        description = request.form['description']
        status = 0  # Default to incomplete
        with sqlite3.connect(DATABASE) as conn:
            cursor = conn.cursor()
            cursor.execute("INSERT INTO tasks (description, status) VALUES (?, ?)", (description, status))
        return redirect(url_for('home'))
    return render_template('create.html')


@app.route('/update/<int:task_id>', methods=['GET', 'POST'])
def update(task_id):
    """Update an existing task."""
    with sqlite3.connect(DATABASE) as conn:
        cursor = conn.cursor()
        if request.method == 'POST':
            description = request.form['description']
            status = int(request.form['status'])
            cursor.execute("UPDATE tasks SET description = ?, status = ? WHERE id = ?", (description, status, task_id))
            return redirect(url_for('home'))
        else:
            cursor.execute("SELECT * FROM tasks WHERE id = ?", (task_id,))
            task = cursor.fetchone()
    return render_template('update.html', task=task)


@app.route('/delete/<int:task_id>', methods=['GET', 'POST'])
def delete(task_id):
    """Delete a task."""
    if request.method == 'POST':
        with sqlite3.connect(DATABASE) as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM tasks WHERE id = ?", (task_id,))
        return redirect(url_for('home'))
    return render_template('delete.html', task_id=task_id)


if __name__ == '__main__':
    init_db()
    app.run(debug=True)
