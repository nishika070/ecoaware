-

# EcoAware Project – Git & Collaboration Guide

This guide explains **how our team will use Git and GitHub to work together safely without losing code or overwriting each other's work**.

Team members must follow these steps whenever they work on the project.

---

# 1. Tools Required

Make sure the following are installed on your system:

1. **Git**
2. **VS Code**
3. **GitHub account**
4. **Node.js** (for frontend later)
5. **Python** (for backend)

Check Git installation:

```
git --version
```

If a version appears, Git is installed.

---

# 2. Cloning the Repository (First Step for Team Members)

Each team member must **clone the repository** to their local machine.

Run this command in terminal:

```
git clone https://github.com/nishika070/ecoaware.git
```

This downloads the entire project.

After cloning, move into the project folder:

```
cd ecoaware
```

Open the project in VS Code:

```
code .
```

Or open manually:

```
VS Code → File → Open Folder → ecoaware
```

---

# 3. Project Folder Structure

Our project is organized as follows:

```
ecoaware

frontend
backend
datasets
docs
```

Explanation:

**frontend**
React website code (UI, pages, components)

**backend**
Python code (API, data fetching, data cleaning)

**datasets**
Stored datasets used for analysis

**docs**
Documentation such as guides and project explanation

---

# 4. Always Pull Before Starting Work

Before writing any code, always download the latest updates.

Run:

```
git pull
```

Why this is important:

If another teammate pushed new code, this command **updates your project** so you don’t overwrite their work.

---

# 5. Making Changes

You can now edit files, create new pages, or write code.

Example tasks:

* Create React components
* Add backend API
* Write data cleaning scripts
* Add graphs or charts

---

# 6. Checking Your Changes

Before committing, you can check what changed.

Run:

```
git status
```

This command shows:

* Modified files
* New files
* Files ready to commit

---

# 7. Adding Files to Git

After making changes, you must tell Git to track them.

Run:

```
git add .
```

Meaning:

Add **all changed files** to the staging area.

You can also add specific files:

```
git add filename.js
```

---

# 8. Committing Changes

A commit is a **snapshot of your changes**.

Run:

```
git commit -m "added AQI dashboard page"
```

Commit messages must explain **what was changed**.

Good examples:

```
added AQI trend graph
fixed navbar layout
added temperature forecast chart
implemented pollution hotspot map
```

Bad examples:

```
update
changes
done
```

Always write meaningful commit messages.

---

# 9. Pushing Changes to GitHub

After committing, upload your code to GitHub.

Run:

```
git push
```

This sends your changes to the remote repository.

Now your teammates can see your code.

---

# 10. How Team Members Get Your Updates

Whenever someone pushes code, others must run:

```
git pull
```

This updates their local project with the newest changes.

---

# 11. Full Daily Workflow

Every developer must follow this sequence:

```
git pull
↓
Write or update code
↓
git status
↓
git add .
↓
git commit -m "describe your change"
↓
git push
```

---

# 12. Important Team Rules

Always follow these rules:

1. Always run **git pull before coding**.
2. Do not modify another teammate’s files without informing them.
3. Write clear commit messages.
4. Test your code before pushing.
5. Push small changes regularly instead of huge changes at once.

---

# 13. Viewing Commit History

You can see past commits with:

```
git log
```

This shows:

* commit history
* author
* changes

---

# 14. Common Problems and Fixes

### Nothing to commit

If you see:

```
nothing to commit
```

It means no new changes were detected.

Check using:

```
git status
```

---

### Pull conflicts

If two people edit the same file, Git may show a conflict.

Solution:

1. Open the conflicting file
2. Fix the conflict manually
3. Save the file
4. Run

```
git add .
git commit -m "resolved merge conflict"
```

---

# 15. Recommended Work Distribution

Example team roles:

### Developer 1 – Frontend

* Homepage dashboard
* Navigation bar
* UI components

### Developer 2 – Air Quality Page

* AQI graphs
* PM2.5 and PM10 charts
* Pollution trends

### Developer 3 – Backend

* Data APIs
* Data cleaning scripts
* Database connection

### Developer 4 – Insights Page

* Pollution hotspot map
* Government insights
* Correlation analysis

---

# 16. Example Project Workflow

```
Frontend developer builds UI
↓
Backend developer provides API
↓
Data developer processes datasets
↓
Charts and maps visualize results
```

---

# 17. Final Goal

The EcoAware system will include:

* Real-time AQI monitoring
* Temperature analysis
* Pollution hotspot visualization
* Health advisory recommendations
* Government policy insights

---

End of Guide.
