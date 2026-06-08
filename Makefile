.PHONY: help install pipeline api dashboard test up down clean

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-12s\033[0m %s\n",$$1,$$2}'

install: ## Create venv and install dependencies
	python3.12 -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt

pipeline: ## Run the full pipeline: download data -> train -> evaluate -> save model + figures
	python -m src.train

api: ## Run the FastAPI service locally (port 8000)
	uvicorn api.main:app --reload --port 8000

dashboard: ## Run the Streamlit dashboard locally (port 8501)
	API_URL=http://localhost:8000 streamlit run dashboard/app.py

test: ## Run the test suite
	pytest -q

up: ## Build and run API + dashboard with Docker Compose
	docker compose up --build

down: ## Stop the Docker Compose stack
	docker compose down

clean: ## Remove caches and generated artifacts (keeps committed model)
	rm -rf __pycache__ */__pycache__ .pytest_cache
