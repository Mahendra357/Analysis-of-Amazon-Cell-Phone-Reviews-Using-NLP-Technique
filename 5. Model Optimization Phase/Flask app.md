## **Overview:**
This Flask application is a simple web-based task management system. It interacts with an SQLite database to store and manage tasks, where each task has:
1. A description.
2. A status (whether it's completed or incomplete).

The system provides a user interface where users can:
- View tasks
- Create new tasks
- Update existing tasks
- Delete tasks

---

## **Detailed Code Breakdown:**

### **1. Importing Necessary Libraries**
```python
from flask import Flask, render_template, request, redirect, url_for
import sqlite3
```
- **Flask**: A web framework to create the web application.
- **render_template**: Used to render HTML templates for the views.
- **request**: To handle incoming data from the user (e.g., form submissions).
- **redirect, url_for**: To handle redirects to different routes (pages) in the app.
- **sqlite3**: Used for connecting and interacting with the SQLite database.

---

### **2. Setting Up the Flask App**
```python
app = Flask(__name__)
DATABASE = 'tasks.db'
```
- **Flask(__name__)**: Initializes the Flask application.
- **DATABASE**: Specifies the SQLite database file (`tasks.db`) that stores the tasks. If it doesn't exist, it will be created automatically.

---

### **3. Initializing the Database**
```python
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
```
- **init_db()**: A function that creates a **`tasks`** table in the SQLite database if it doesn't already exist.
    - **id**: A unique identifier for each task, automatically increments.
    - **description**: A text field that stores the description of the task.
    - **status**: An integer field that stores the task's completion status (0 for incomplete, 1 for complete).

---

### **4. Displaying All Tasks**
```python
@app.route('/')
def home():
    """Display all tasks."""
    with sqlite3.connect(DATABASE) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM tasks")
        tasks = cursor.fetchall()
    return render_template('home.html', tasks=tasks)
```
- **home()**: The home route (`'/'`) is accessed when the user visits the app’s main page.
    - It connects to the SQLite database, fetches all tasks, and passes them to the `home.html` template for rendering.
    - **`tasks`**: A list of all tasks from the database is displayed on the home page.

---

### **5. Creating a New Task**
```python
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
```
- **create()**: Handles the task creation process.
    - When the user visits the `/create` page (via **GET**), it renders the `create.html` form.
    - When the user submits the form (via **POST**), the task description is saved in the database with a default **status of 0** (incomplete).
    - After saving the task, it redirects the user back to the home page (`'home'`).

---

### **6. Updating an Existing Task**
```python
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
```
- **update()**: This route allows users to update an existing task.
    - **GET request**: Fetches the task from the database using the `task_id` provided in the URL, and renders the `update.html` template with the current task details.
    - **POST request**: Updates the task’s description and status in the database, then redirects to the home page.

---

### **7. Deleting a Task**
```python
@app.route('/delete/<int:task_id>', methods=['GET', 'POST'])
def delete(task_id):
    """Delete a task."""
    if request.method == 'POST':
        with sqlite3.connect(DATABASE) as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM tasks WHERE id = ?", (task_id,))
        return redirect(url_for('home'))
    return render_template('delete.html', task_id=task_id)
```
- **delete()**: Allows users to delete a task.
    - **GET request**: Displays the confirmation page to delete the task.
    - **POST request**: Deletes the task with the specified `task_id` from the database and redirects the user back to the home page.

---

### **8. Running the App**
```python
if __name__ == '__main__':
    init_db()
    app.run(debug=True)
```
- **if __name__ == '__main__'**: Ensures that the `init_db()` function is called to initialize the database when the app starts.
- **app.run(debug=True)**: Runs the Flask application in **debug mode** so that it reloads automatically when code changes. The app will be accessible on `localhost:5000`.

---

## **Application Flow:**

1. **Home Page (`/`)**: Displays all tasks.
2. **Create Task (`/create`)**: Allows the user to add a new task.
3. **Update Task (`/update/<task_id>`)**: Allows the user to update the details of a task.
4. **Delete Task (`/delete/<task_id>`)**: Allows the user to delete a task from the database.

---

### **Technologies Used:**
- **Flask**: Web framework for building the application.
- **SQLite**: Lightweight database used to store tasks.
- **HTML Templates**: Used for rendering views for displaying and interacting with tasks.

---

### **Summary:**

This Flask application provides a simple web interface for managing tasks, allowing users to:
- View, create, update, and delete tasks.
- The tasks are stored in an SQLite database, where each task has a description and a status indicating whether it is completed.
- The application uses **Flask routes** to handle the different operations (CRUD), and **SQLite queries** to interact with the database.
