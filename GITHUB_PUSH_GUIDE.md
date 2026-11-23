# GitHub Push Guide — buttercookiegh (macOS / uv)

This project folder is prepared, but it has **not** been uploaded to GitHub yet. Follow these steps on your Mac to publish it and obtain the required commit hash.

## 1. Unzip and open in VS Code

Download and unzip `rng-entropy-ml-buttercookiegh.zip`. The folder inside is `rng-entropy-ml`. Move it to Documents and open it in VS Code. Open **Terminal → New Terminal**. Then:

```bash
cd ~/Documents/rng-entropy-ml
ls -a
```

Confirm you see `README.md`, `pyproject.toml`, `uv.lock`, `data/` and `scripts/`.

## 2. Set up uv and verify dataset preparation

If uv is missing on your Mac, install it with `brew install uv` (if Homebrew is installed). Then:

```bash
uv sync
uv run python scripts/prepare_data.py
```

You should see 6,000 sequences and split totals of 403,200 training windows and 100,800 testing windows. Model training is **not required for this dataset-selection assignment**.

## 3. Configure Git identity (only once per Mac)

```bash
git config --global user.name "buttercookiegh"
git config --global user.email "israt.bushra.eee@gmail.com"
git config --global init.defaultBranch main
```

Note: `main` must be a standalone word; `mainthis` would be an incorrect branch setting. This config sets commit author information; it does **not** log you in to GitHub. GitHub may require that email to be verified on your account to attribute commits.

## 4. Create an empty repository on GitHub

Open: https://github.com/new

- Owner: **buttercookiegh**
- Repository name: **rng-entropy-ml**
- Visibility: **Public** so the instructor can open the link directly, or **Private** if you will grant the instructor access
- Do **not** check `Add a README file`, `.gitignore`, or license

Click **Create repository**.

## 5. Commit and push from your Mac terminal

Run inside `~/Documents/rng-entropy-ml`:

```bash
git init
git branch -M main
git add .
git status
git commit -m "Initial commit: RNG dataset README and uv setup"
git remote add origin https://github.com/buttercookiegh/rng-entropy-ml.git
git push -u origin main
```

When prompted to authenticate, use GitHub's approved browser/credential flow or a personal access token; your GitHub account password cannot be used directly for HTTPS Git operations.

If `origin` already exists, use `git remote set-url origin https://github.com/buttercookiegh/rng-entropy-ml.git` instead of adding it again.

## 6. Find the professor's requested commit hash

```bash
git rev-parse HEAD
git status
```

Submit the exact output of `git rev-parse HEAD`, not a placeholder. Check these links **after a successful push**:

- Repository: https://github.com/buttercookiegh/rng-entropy-ml
- README: https://github.com/buttercookiegh/rng-entropy-ml/blob/main/README.md
- Version-specific README: `https://github.com/buttercookiegh/rng-entropy-ml/blob/YOUR_FULL_COMMIT_HASH/README.md`

## Optional: GitHub CLI (creates and pushes in one command)

You can use GitHub CLI instead of Steps 4 and 5, **after** local `git init`, `git add .`, and `git commit`:

```bash
brew install gh
gh auth login
gh repo create buttercookiegh/rng-entropy-ml --public --source=. --remote=origin --push
```

This CLI command creates a **public** GitHub repository. Use `--private` if you want a private repository and intend to grant your instructor access. Do not run this creation command if you already created the remote repository in Step 4.

## Files to commit

Include: `README.md`, `pyproject.toml`, `uv.lock`, `.python-version`, `.gitignore`, `scripts/`, `data/`.

Do not commit the `.venv/` directory. The provided `.gitignore` already excludes it.
