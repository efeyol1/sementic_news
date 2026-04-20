FROM python:3.12-slim

WORKDIR /app

# Sistem bağımlılıkları (transformers için gerekli)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml .
RUN pip install --no-cache-dir -e ".[dev]" \
    && pip install --no-cache-dir uvicorn[standard]

COPY src/ src/
COPY data/analyzed/ data/analyzed/

EXPOSE 8000

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
