# CONTRIBUTING.md

> A beginner‑friendly guide for this repo. If any step doesn’t match your setup, ping us in a new issue and we’ll adjust.

---

## 1) Dev environment

We develop and test on **Python ≥ 3.10** with a local virtual environment. Optional tools we use: **pytest**, **ruff**, **black**, **mypy**, and **pre-commit**.

### Quick setup (cross‑platform)

```bash
# 1) Clone your fork (see §2 for forking)
git clone https://github.com/LSD-Collaboration/Levitated-Sensor-Detector-Experiment.git
cd Levitated-Sensor-Detector-Experiment

# 2) Create and activate a virtual environment
python -m venv .venv          # Windows: py -m venv .venv
source .venv/bin/activate     # Windows: .venv\Scripts\activate

# 3) Install the project in editable mode + dev tools
# Option A (preferred if pyproject exposes [project.optional-dependencies].dev)
pip install -e ".[dev]"
# Option B (fallback)
# pip install -r requirements.txt
# pip install -r requirements-dev.txt

# 4) Install pre-commit hooks (runs format/lint before each commit)
pre-commit install
pre-commit run --all-files   # one-time check of the whole repo
```

**What you just did:**

* *Virtual environment* isolates packages for this project so you don’t pollute your global Python.
* *Editable install* (`-e`) means local code changes are immediately used without reinstalling.
* *Pre-commit* runs quick quality checks automatically when you commit (can be skipped with `--no-verify` in emergencies, but please fix issues instead).

---

## 2) How to add your change (step‑by‑step)

### If you don’t have write access:

1. **Fork** the repo on GitHub (button in the top‑right).
2. **Clone** your fork locally:

   ```bash
   git clone https://github.com/<your-username>/<repo>.git
   cd Levitated-Sensor-Detector-Experiment
   git remote add upstream https://github.com/LSD-Collaboration/Levitated-Sensor-Detector-Experiment.git
   ```
3. **Create a branch** for your change (never work on `main`):

   ```bash
   git checkout -b feat/better-iter-chunks
   ```
4. **Make edits** in your editor.
5. **Run checks** locally:

   ```bash
   pre-commit run --all-files
   pytest -q
   ```
6. **Stage** and **commit** your changes:

   ```bash
   git add -A              # stage all modified files
   git commit -m "feat(iter): add overlap example to docs"
   ```
7. **Sync** with upstream `main` (keep your branch up to date):

   ```bash
   git fetch upstream
   git pull --rebase upstream main   # see §6 for what rebase means
   ```
8. **Push** your branch to *your fork*:

   ```bash
   git push -u origin HEAD
   ```
9. **Open a Pull Request** on GitHub (see §7).

### If you do have write access

* Same steps, but you can clone the upstream repo directly and push branches to it. Still create feature branches and open PRs (don’t push to `main`).

**Jargon explained:** *fork* = your copy of the repo on GitHub. *branch* = a parallel line of work. *stage* = tell Git which changes to include in the next commit. *commit* = a saved checkpoint with a message.

---

## 3) Make your change (practical guidance)

* Keep the change **small and focused**. Separate refactors from feature work.
* Follow the module’s style:

  * **NumPy‑style docstrings**
  * **Type hints** everywhere
  * Use **`logging`**, not `print`
  * Avoid hidden global state; pass values explicitly
  * Use clear, stable **names** (match existing patterns)
* Prefer **pure functions** where possible (easy to test). Break long functions into smaller helpers.
* Add comments that explain **why**, not what the code already makes obvious.
* If behavior is subtle (e.g., chunk boundaries, time bases), put an **example** in the docstring.

**Helpful commands:**

```bash
ruff check .         # fast linter
black .              # format code
mypy tdms_tools.py   # type check
pytest -q            # run tests
```

---

## 4) Tests (how, why, when)

**Why test?** To prove your change works and keep it working as the code evolves.

**When to test:**

* Any new feature or bug fix
* Edge cases (file boundaries, `drop_incomplete=False`, `absolute_time=True/False`)
* Performance‑critical paths (e.g., `iter_chunks`)—at least a smoke test

**How to run tests:**

```bash
pytest -q
pytest -q tests/test_iter.py::test_overlap_logic   # run a single test
pytest --cov=tdms_tools --cov-report=term-missing  # coverage
```

**How to write a test (example):**

```python
# tests/test_time_axis.py
import numpy as np
from tdms_tools import TDMSDataset
from nptdms import TdmsWriter, ChannelObject

def test_read_window_rel(tmp_path):
    # Create a tiny TDMS file
    p = tmp_path / "mini.tdms"
    with TdmsWriter(p) as w:
        w.write_segment([ChannelObject("group", "wfg", np.arange(10, dtype=float))])
    ds = TDMSDataset(tmp_path)
    ds.build_index(["wfg"])
    t_rel, y, _ = ds.read_window_rel("wfg", 0.0, 0.00009)
    assert y.size > 0
```

*(real tests in this repo may use helpers/fixtures; this is just a sketch.)*

---

## 5) Commits (what to write and why it matters)

A good commit message tells **what changed** and **why**. We use a light version of **Conventional Commits**:

```
<type>(<scope>): <short summary>

<body explaining the WHY>
```

**Types:** `feat`, `fix`, `docs`, `refactor`, `test`, `build`, `ci`.

**Examples:**

* `feat(iter): add tail_fill="nan" option`
* `fix(zarr): correct time dtype for absolute writes`
* `docs(guide): expand Hello, chunks`

**Tips:**

* Keep summaries ≤ 72 chars
* Reference issues: `Fixes #123` (auto‑closes on merge)
* Make separate commits for unrelated changes

---

## 6) Rebase & Push (in depth, beginner‑friendly)

**What is rebase?**
Rebase = “replay my commits on top of the latest `main` as if I started there.” It keeps history linear and avoids noisy “merge commits.”

**Why do it?**

* Resolve conflicts early
* Make PRs easier to review

**How to rebase your branch onto upstream ****************`main`****************:**

```bash
# Make sure you have the upstream remote
git remote -v                           # should show "upstream" and "origin"

# Fetch the latest changes from upstream
git fetch upstream

# Move (replay) your branch commits on top of upstream/main
git rebase upstream/main
```

During a rebase, Git may stop on conflicts. You’ll see the files that conflict:

```bash
git status
# edit files to resolve conflicts, then:
git add <file1> <file2>
git rebase --continue
```

If you get stuck:

```bash
git rebase --abort   # go back to how things were before the rebase
```

**Pushing after a rebase:**
Because history changed, use a *safe* force push:

```bash
git push --force-with-lease
```

This only overwrites your remote branch if no one else pushed in the meantime.

**Mental model (ASCII):**

```
Before:   main: A---B---C
          you :       \--d---e  (your commits)
Rebase →  you : A---B---C---d'---e'
```

---

## 7) PR (Pull Request) — what it is and how to open one

**What is a PR?** A *Pull Request* asks maintainers to pull your branch into the main project. It’s a place for review, discussion, and automated checks.

**Open a PR:**

1. Push your branch (`git push -u origin HEAD`).
2. On GitHub, you’ll see a prompt to “Open pull request.” Click it.
3. Fill out the template:

   * **Title:** concise summary (use the same wording as your best commit)
   * **Description:** what changed and *why* (link issues; include “Fixes #123” if applicable)
   * **How to test:** commands or steps reviewers can run
4. Ensure checks pass (CI, lint, tests). Fix anything red.
5. Request review if needed. Be responsive to comments; push updates as new commits (don’t rewrite history mid‑review unless asked).
6. Once approved: maintainers will **squash & merge** (preferred) or ask you to rebase for a clean history.

**After merge:**

* Delete your branch on GitHub
* Update local `main`:

  ```bash
  git checkout main
  git fetch upstream
  git pull --rebase upstream main
  ```

---

## Troubleshooting quickies

* **Hook failures:** run `pre-commit run --all-files` and follow the tool messages.
* **Windows path issues:** prefer raw strings like `r"C:\data\run01"` in examples.
* **Large files:** don’t commit data; use `.gitignore` and document how to fetch/generate.

Happy hacking! 🎉
