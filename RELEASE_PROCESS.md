# Release Process (Minimal)

## 1) Pre-release checks

Run locally:

```bash
/Users/jiangshengyu/Documents/program/python/Python_Coding_Agent/.venv/bin/python -m py_compile evolve.py main.py agent/code_generator.py
/Users/jiangshengyu/Documents/program/python/Python_Coding_Agent/.venv/bin/python test_agent.py
/Users/jiangshengyu/Documents/program/python/Python_Coding_Agent/.venv/bin/python test_memory_agent.py
/Users/jiangshengyu/Documents/program/python/Python_Coding_Agent/.venv/bin/python test_reflection_agent.py
```

## 2) Prepare release commit

```bash
git add -A
git commit -m "release: <version>"
```

## 3) Tag and push

Use semantic tags (example: `v0.1.0`):

```bash
git tag -a v0.1.0 -m "Release v0.1.0"
git push origin main
git push origin v0.1.0
```

## 4) Verify in GitHub

- Confirm `CI` workflow is green on `main`.
- Create GitHub Release from tag `vX.Y.Z`.
- Include highlights and risk notes from `data/evolution/STABILITY_CHANGELOG.md`.

## 5) Rollback (if needed)

Rollback by reverting release commit (safer than force-push):

```bash
git revert <release_commit_sha>
git push origin main
```

## 6) Recommended repo settings

In GitHub branch protection for `main`:

- Require pull request before merge.
- Require status checks to pass (`CI / test`).
- Restrict force pushes.
