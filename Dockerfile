# === Stage 1: Builder ===
# Pinned by digest for reproducible builds and supply-chain integrity (OVH-061).
# Dependabot (docker ecosystem) bumps the tag+digest on a schedule. To bump
# manually: `docker pull python:3.13-slim && docker inspect --format '{{index .RepoDigests 0}}' python:3.13-slim`.
FROM python:3.13-slim@sha256:9d2e5553305c7c7b0097999bb17187c69b921ccd6bc9d40e4bb5ebe652c00285 AS builder

ENV PYTHONDONTWRITEBYTECODE=1

WORKDIR /build

# Build backend pinned by hash (TW-AUD-033): hatchling and its dependencies come
# from requirements-build.txt, not from whatever PyPI serves at build time. It is
# installed into the builder's base interpreter, not the runtime venv, so the
# image ships no build tooling. The base image's own pip is used as-is: it is
# part of the digest-pinned input above, so it is not upgraded here.
COPY requirements-build.txt ./
RUN pip install --no-cache-dir --require-hashes -r requirements-build.txt

COPY pyproject.toml README.md requirements.txt ./
COPY app/ ./app/
RUN pip wheel --no-cache-dir --no-deps --no-build-isolation --wheel-dir /build/dist .

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --no-cache-dir --require-hashes -r requirements.txt \
 && pip install --no-cache-dir --no-deps /build/dist/*.whl

# === Stage 2: Runtime ===
# Same digest pin as the builder stage (OVH-061).
FROM python:3.13-slim@sha256:9d2e5553305c7c7b0097999bb17187c69b921ccd6bc9d40e4bb5ebe652c00285

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH"

WORKDIR /app

# Copy the virtual environment from builder
COPY --from=builder /opt/venv /opt/venv

# gosu for privilege de-escalation in the entrypoint (drops from root to the
# host-aligned PUID/PGID after fixing volume ownership). Fetched from its GitHub
# release and verified against a pinned per-arch SHA-256 (TW-AUD-033): the
# Debian package was the one build input resolved outside the declared locks,
# and apt cannot pin a single version across both published architectures.
# To bump: change GOSU_VERSION and both sums from the release's SHA256SUMS file
# (signed; verify with tianon's key B42F6819007F00F88E364FD4036A9C25BF357DD4).
# License text: THIRD_PARTY_NOTICES.md (shipped below).
ARG TARGETARCH
ARG GOSU_VERSION=1.19
ADD https://github.com/tianon/gosu/releases/download/${GOSU_VERSION}/gosu-${TARGETARCH} /usr/local/bin/gosu
RUN set -eu; \
    case "$TARGETARCH" in \
      amd64) sum="52c8749d0142edd234e9d6bd5237dff2d81e71f43537e2f4f66f75dd4b243dd0" ;; \
      arm64) sum="3a8ef022d82c0bc4a98bcb144e77da714c25fcfa64dccc57f6aba7ae47ff1a44" ;; \
      *) echo "unsupported TARGETARCH: $TARGETARCH" >&2; exit 1 ;; \
    esac; \
    echo "$sum  /usr/local/bin/gosu" | sha256sum -c -; \
    chmod 0755 /usr/local/bin/gosu; \
    gosu nobody true

# Copy application code and example config
COPY app/ ./app/
COPY config.example.yml ./

# Ship license text for this project and for the vendored front-end assets in
# app/static/vendor/ (Pico CSS, htmx) alongside the image that redistributes
# them (AUG-341).
COPY LICENSE THIRD_PARTY_NOTICES.md /usr/share/licenses/topic-watch/

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
