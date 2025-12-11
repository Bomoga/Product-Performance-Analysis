# GitHub Repository Setup Guide

Follow these steps to create a GitHub repository for this project.

## Step 1: Initialize Git Repository

Open PowerShell or Command Prompt in the project directory and run:

```bash
git init
```

## Step 2: Add All Files

```bash
git add .
```

## Step 3: Create Initial Commit

```bash
git commit -m "Initial commit: ML Supermarket Analysis Application"
```

## Step 4: Create GitHub Repository

1. Go to [GitHub.com](https://github.com) and sign in
2. Click the **+** icon in the top right corner
3. Select **New repository**
4. Fill in the repository details:
   - **Repository name**: `ml-supermarket-analysis` (or your preferred name)
   - **Description**: "Machine Learning application for supermarket product sales analysis with K-means clustering and regression"
   - **Visibility**: Choose Public or Private
   - **DO NOT** initialize with README, .gitignore, or license (we already have these)
5. Click **Create repository**

## Step 5: Connect Local Repository to GitHub

After creating the repository, GitHub will show you commands. Use these:

```bash
# Add the remote repository (replace YOUR_USERNAME and REPO_NAME)
git remote add origin https://github.com/YOUR_USERNAME/REPO_NAME.git

# Rename default branch to main (if needed)
git branch -M main

# Push your code to GitHub
git push -u origin main
```

## Step 6: Verify

1. Refresh your GitHub repository page
2. You should see all your files uploaded
3. The README.md should display automatically

## Optional: Add Repository Topics

On your GitHub repository page:
1. Click the gear icon ⚙️ next to "About"
2. Add topics like: `machine-learning`, `kmeans`, `regression`, `python`, `tkinter`, `data-analysis`

## Optional: Add a License

If you want to add a license:
1. Create a `LICENSE` file in the root directory
2. Choose an appropriate license (MIT, Apache 2.0, etc.)
3. Commit and push it

## Future Updates

When you make changes to the code:

```bash
# Check what files changed
git status

# Add changed files
git add .

# Commit with a descriptive message
git commit -m "Description of your changes"

# Push to GitHub
git push
```

## Troubleshooting

### If you get authentication errors:
- Use GitHub Personal Access Token instead of password
- Or set up SSH keys for authentication

### If you want to exclude data files:
- Uncomment the data exclusion lines in `.gitignore`
- This prevents large CSV files from being uploaded

### If you want to add collaborators:
1. Go to repository Settings
2. Click "Collaborators"
3. Add GitHub usernames or email addresses

