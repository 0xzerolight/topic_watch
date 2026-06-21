# === Stage 1: Builder ===
# Pinned by digest for reproducible builds and supply-chain integrity (OVH-061).
# Dependabot (docker ecosystem) bumps the tag+digest on a schedule. To bump
# manually: `docker pull python:3.11-slim && docker inspect --format '{{index .RepoDigests 0}}' python:3.11-slim`.
FROM python:3.11-slim@sha256:ae52c5bef62a6bdd42cd1e8dffef86b9cd284bde9427da79839de7a4b983e7ca AS builder

ENV PYTHONDONTWRITEBYTECODE=1

WORKDIR /build

# Install build dependencies if needed (none currently, but future-proof)
RUN pip install --no-cache-dir --upgrade pip

# Copy project metadata and install into a venv
COPY pyproject.toml README.md requirements.txt ./
COPY app/ ./app/

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --no-cache-dir --require-hashes -r requirements.txt && pip install --no-cache-dir --no-deps .

# === Stage 2: Runtime ===
# Same digest pin as the builder stage (OVH-061).
FROM python:3.11-slim@sha256:ae52c5bef62a6bdd42cd1e8dffef86b9cd284bde9427da79839de7a4b983e7ca

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH"

WORKDIR /app

# Copy the virtual environment from builder
COPY --from=builder /opt/venv /opt/venv

# Install gosu for privilege de-escalation in the entrypoint (drops from root
# to the host-aligned PUID/PGID after fixing volume ownership).
RUN apt-get update && \
    apt-get install -y --no-install-recommends gosu && \
    rm -rf /var/lib/apt/lists/* && \
    gosu nobody true

# Copy application code and example config
COPY app/ ./app/
COPY config.example.yml ./

# Create the runtime user/group. UID/GID 1000 is the default; the entrypoint
# remaps these to the host-provided PUID/PGID at startup so bind-mounted ./data
# is writable regardless of the host user's UID.
RUN groupadd --gid 1000 appgroup && \
    useradd --uid 1000 --gid appgroup --create-home appuser && \
    mkdir -p /app/data && \
    chown -R appuser:appgroup /app/data

# Entrypoint runs as root, chowns the data volume to PUID/PGID, then drops
# privileges with gosu so the app itself runs unprivileged.
COPY docker-entrypoint.sh /docker-entrypoint.sh
RUN chmod +x /docker-entrypoint.sh

LABEL org.opencontainers.image.source="https://github.com/0xzerolight/topic_watch"
LABEL org.opencontainers.image.description="Self-hosted news monitoring with AI-powered novelty detection"
LABEL org.opencontainers.image.licenses="GPL-3.0-or-later"

EXPOSE 8000

# OVH-122: explicit urlopen timeout (< the 5s Docker --timeout) so the probe is
# self-bounded if /health hangs, rather than relying on Docker's SIGKILL alone.
HEALTHCHECK --interval=30s --timeout=5s --retries=3 --start-period=10s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=4)"

STOPSIGNAL SIGTERM

ENTRYPOINT ["/docker-entrypoint.sh"]
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
