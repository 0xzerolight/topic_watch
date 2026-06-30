# Releasing

topic_watch ships as a Docker image on GHCR. A release is a git tag; CI does the
rest. There is no PyPI package and no manual image build.

## Versioning

- Single source of truth: `version` in `pyproject.toml`. Nothing else.
  `app/__init__.py` reads it at runtime via importlib metadata, so you never edit
  a version string in Python.
- Semantic Versioning `MAJOR.MINOR.PATCH`:
  - PATCH (x.y.Z): bug fixes only.
  - MINOR (x.Y.0): new backward-compatible features (look at `[Unreleased]` in
    CHANGELOG.md, anything under `### Added`/`### Changed` that isn't breaking).
  - MAJOR (X.0.0): breaking changes (config/schema/API the user must act on).
- Git tag is the version prefixed with `v` (e.g. `v1.2.0`).

## Release steps

All on `main`, fully merged and green (`make ci` passes).

1. Decide the new version from what's under `## [Unreleased]` in CHANGELOG.md.

2. Bump `pyproject.toml`:

       version = "X.Y.Z"

3. Promote the CHANGELOG. Rename `## [Unreleased]` to `## [X.Y.Z] - YYYY-MM-DD`
   (today's date), then add a fresh empty `## [Unreleased]` block above it. Drop
   empty `### Added/Changed/Fixed/Security` subsections.

4. Sanity check the build locally (optional but cheap):

       make ci

5. Commit:

       git add pyproject.toml CHANGELOG.md
       git commit -m "chore: release vX.Y.Z"

6. Tag and push:

       git push origin main
       git tag vX.Y.Z
       git push origin vX.Y.Z

7. CI (`.github/workflows/docker-publish.yml`) builds multi-arch (amd64/arm64)
   and pushes to `ghcr.io/0xzerolight/topic_watch` with tags `latest`, `X.Y.Z`,
   and `X.Y`.

## Verify

- Watch the run: `gh run watch` (or the Actions tab).
- Confirm the image:

      docker pull ghcr.io/0xzerolight/topic_watch:X.Y.Z

- Confirm the GitHub tag exists and `latest` moved to the new digest.

## Upgrading a deployment (for users / your own box)

    cd ~/topic-watch
    docker compose pull && docker compose up -d

Pin a specific release with `TOPIC_WATCH_REF=vX.Y.Z` — the install scripts honor
it. The DB is auto-backed-up before any schema migration.

## Notes

- Pushing to `main` runs CI only and publishes no image. Images are published
  solely by a `v*` tag (`latest`, `X.Y.Z`, `X.Y`). Always tag for a real release.
