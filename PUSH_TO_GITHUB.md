# How to push this to GitHub

Run these commands in your terminal, inside the project folder.

## Step 1 — Copy these files into your existing project folder

Copy everything from this repo structure into your local project folder.

## Step 2 — Open terminal in your project folder

```bash
cd path/to/your/project
```

## Step 3 — Initialise git (if not already done)

```bash
git init
git config user.email "your@email.com"
git config user.name "Farhan Hashmi"
```

## Step 4 — Connect to your GitHub repo

```bash
git remote add origin https://github.com/Shaunfarhan16/Large-Scale-Airbnb-Price-Prediction-Spatial-Analytics-Platform.git
```

If remote already exists:
```bash
git remote set-url origin https://github.com/Shaunfarhan16/Large-Scale-Airbnb-Price-Prediction-Spatial-Analytics-Platform.git
```

## Step 5 — Stage, commit, and push

```bash
git add .
git status          # review what will be committed
git commit -m "Refactor: modular src pipeline, leakage-safe scaler, docstrings, README"
git push -u origin main
```

If your branch is called master:
```bash
git push -u origin master
```

## What gets committed (safe)
- README.md
- requirements.txt
- .gitignore
- src/*.py (all fixed scripts)
- app/AirbnbDashboard.py
- notebooks/*.ipynb
- models/.gitkeep
- figures/.gitkeep
- data/.gitkeep

## What does NOT get committed (excluded by .gitignore)
- data/Airbnb_clean.csv (152MB — too large)
- data/Airbnb_london.csv (raw data)
- models/lgbm_pipeline.pkl (binary — use Git LFS if needed)
- figures/*.png (generated at runtime)
- __pycache__/
- .env
