FROM python:3.12-slim

WORKDIR /app

# Single source of truth = pyproject.toml. Copying src/ before install
# because hatchling needs the package tree present to build the wheel.
COPY pyproject.toml ./
COPY src/ src/
RUN pip install --no-cache-dir .

COPY data/analyzed/ data/analyzed/

EXPOSE 8000

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
