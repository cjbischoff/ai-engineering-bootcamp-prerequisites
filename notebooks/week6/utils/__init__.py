"""
Week 6 notebook support package (Sprint 5).

Re-exports nothing by default: notebooks import concrete helpers from
`utils.utils` and `utils.tools` after `sys.path` includes `notebooks/week6`.

Why a package: keeps the same `from utils.X import Y` pattern as Week 5 while
living beside `01-litellm-router.ipynb` for learners who open the repo at root.
"""
