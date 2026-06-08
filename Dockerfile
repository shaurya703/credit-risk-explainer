# Shared image for both the API and the dashboard (compose overrides the command).
FROM python:3.12-slim

# libgomp1 is the OpenMP runtime XGBoost needs at import time.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1

# Install dependencies first for better layer caching.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App code + the trained model artifact + pre-rendered figures.
COPY src/ ./src/
COPY api/ ./api/
COPY dashboard/ ./dashboard/
COPY models/ ./models/
COPY reports/ ./reports/

EXPOSE 8000 8501

# Default to the API; the dashboard service overrides this in docker-compose.
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
