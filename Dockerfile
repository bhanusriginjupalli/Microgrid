# Stage 1: Build React + Vite frontend bundle
FROM node:22-alpine AS frontend-builder
WORKDIR /app/frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci --legacy-peer-deps
COPY frontend/ ./
RUN npm run build

# Stage 2: FastAPI + Simulation + ML runtime
FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg \
    PORT=8000

# libgomp1 is required by LightGBM; curl is useful for container healthchecks
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt ./
# Install CPU-only PyTorch wheel first to avoid ~2GB CUDA libraries in cloud containers,
# then install the remaining Python dependencies.
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.15.0 \
    && grep -vE '^(torch==|jupyter$|pytest==)' requirements.txt > /tmp/requirements-runtime.txt \
    && pip install --no-cache-dir -r /tmp/requirements-runtime.txt \
    && rm -f /tmp/requirements-runtime.txt

COPY . .
COPY --from=frontend-builder /app/frontend/dist ./frontend/dist

EXPOSE 8000

CMD ["sh", "-c", "uvicorn backend.app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
