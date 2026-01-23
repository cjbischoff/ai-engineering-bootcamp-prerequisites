# Makefile - Convenient commands for common development tasks
#
# What is Make?
#   Make is a build automation tool that runs commands defined in this Makefile.
#   Instead of typing long commands, you type: make <target-name>
#
# How to use:
#   make run-docker-compose    # Start all services
#   make clean-notebook-outputs # Clean Jupyter outputs
#   make run-evals-retriever   # Run RAG evaluation

# Target: run-docker-compose
# Purpose: Start the full application stack (API, UI, Qdrant) in Docker
# When to use: Development, testing, or running the chatbot locally
run-docker-compose:
	# Step 1: Sync dependencies
	# Why? Ensures all required packages are installed before building
	uv sync
	# Step 2: Build and start all Docker containers
	# --build: Rebuild images to pick up code changes
	# This starts: FastAPI backend, Streamlit UI, Qdrant vector DB
	docker compose up --build

# Target: clean-notebook-outputs
# Purpose: Remove all cell outputs from Jupyter notebooks before committing
# Why? Outputs can be large, contain sensitive data, or create merge conflicts
# When to use: ALWAYS before committing notebooks to git
clean-notebook-outputs:
	# Use nbconvert to clear outputs in-place for all notebooks
	# notebooks/**/*.ipynb = all .ipynb files in notebooks/ and subdirectories
	uv run jupyter nbconvert --clear-output --inplace notebooks/**/*.ipynb

# Target: run-evals-retriever
# Purpose: Run RAGAS evaluation metrics against the RAG pipeline
# When to use: After code changes, to ensure quality hasn't regressed
# Output: LangSmith URL with detailed evaluation results
run-evals-retriever:
	# Step 1: Sync dependencies (ensure ragas, langsmith installed)
	uv sync

	# Step 2: Run the evaluation script with proper environment setup
	# Let's break down this complex command:

	# PYTHONPATH configuration (critical for imports to work):
	#   $${PWD}/apps/api/src   - Add api source directory (for 'from api.agents import...')
	#   :$$PYTHONPATH          - Preserve existing PYTHONPATH
	#   :$${PWD}               - Add project root
	#   Why $$? In Makefiles, $ means variable; $$ escapes it to pass to shell

	# uv run --env-file .env  - Run with environment variables from .env
	#   Loads: OPENAI_KEY, LANGSMITH_API_KEY, etc.

	# python apps/api/evals/eval_retriever.py - Execute the evaluation script
	#   Note: We run it as a script (not a module) because it's at apps/api/evals/
	#   which is outside the standard api/ package structure
	PYTHONPATH=$${PWD}/apps/api/src:$$PYTHONPATH:$${PWD} uv run --env-file .env python apps/api/evals/eval_retriever.py

# How this Makefile works:
#
# When you type: make run-evals-retriever
# Make executes each line indented under that target (lines must start with TAB, not spaces!)
#
# Common Makefile gotchas:
#   1. Must use TABS for indentation (spaces cause errors)
#   2. $$ escapes $ for shell variables
#   3. Each line runs in separate shell (can't cd in one line and use it in next)
#   4. No output = success (Make is silent by default)
#
# Adding new targets:
# target-name:
# 	command1
# 	command2
