FROM python:3.12-slim

WORKDIR /app

# Single source of truth = pyproject.toml. Copying src/ before install
# because hatchling needs the package tree present to build the wheel.
COPY pyproject.toml ./
COPY src/ src/
# [serving] adds optimum[onnxruntime] for the /api/predict endpoint.
RUN pip install --no-cache-dir ".[serving]"

# Bake the ONNX int8 sentiment model into the image so /api/predict has
# zero export overhead on cold starts. Re-runs on every container build,
# which means a manual Render redeploy is needed after a weekly retrain
# pushes a new model to HF Hub.
RUN python -m src.serving.onnx_export

COPY data/analyzed/ data/analyzed/

EXPOSE 8000

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
