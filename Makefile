run-docker-compose:
	uv sync
	docker compose up --build

clean-notebook-outputs:
	uv run jupyter nbconvert --clear-output --inplace notebooks/**/*.ipynb