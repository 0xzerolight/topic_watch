# === Stage 1: Builder ===
FROM python:3.11-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1

WORKDIR /build

# Install build dependencies if needed (none currently, but future-proof)
RUN pip install --no-cache-dir --upgrade pip

# Copy project metadata and install into a venv
COPY pyproject.toml README.md ./
COPY app/ ./app/

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --no-cache-dir .

# === Stage 2: Runtime ===
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH"

WORKDIR /app

# Copy the virtual environment from builder
COPY --from=builder /opt/venv /opt/venv

# Copy application code and example config
COPY app/ ./app/
COPY config.example.yml ./

# Create non-root user with write access to data volume
RUN groupadd --system appgroup && \
    useradd --system --gid appgroup --create-home appuser && \
    mkdir -p /app/data && \
    chown -R appuser:appgroup /app/data

USER appuser

LABEL org.opencontainers.image.source="https://github.com/0xzerolight/topic_watch"
LABEL org.opencontainers.image.description="Self-hosted news monitoring with AI-powered novelty detection"
LABEL org.opencontainers.image.licenses="GPL-3.0-or-later"

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --retries=3 --start-period=10s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

STOPSIGNAL SIGTERM

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
