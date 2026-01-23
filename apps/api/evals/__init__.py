"""
Evaluation Scripts Package for RAG Pipeline

This package contains scripts for evaluating the quality of our RAG (Retrieval-Augmented Generation) system.

Purpose:
    - Organize evaluation-related code separately from production code
    - Enable Python to recognize 'evals' as a module (required for imports)
    - Provide a namespace for evaluation utilities

Why __init__.py is needed:
    Without this file, Python won't recognize the 'evals' directory as a package.
    This would cause: ModuleNotFoundError: No module named 'api.evals'

Module Structure:
    api/
    ├── evals/
    │   ├── __init__.py  (this file - makes 'evals' a Python package)
    │   └── eval_retriever.py  (script for evaluating retrieval quality)
    ├── agents/
    │   └── retrieval_generation.py  (the RAG pipeline we're evaluating)
    └── ...

How to run evaluations:
    make run-evals-retriever

What's evaluated:
    - Retrieval Quality: Are we finding the right products?
    - Generation Quality: Are answers accurate and relevant?
    - End-to-End Performance: Does the full RAG pipeline work well?

Results:
    Evaluation results are stored in LangSmith and viewable via the web UI.
"""
