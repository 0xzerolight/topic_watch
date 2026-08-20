"""Static checks for install-script / env-template secret hygiene.

These guard the non-Python deliverables of Task 2.3:
- OVH-063: ``scripts/install.sh`` must lock down the generated ``.env`` (chmod 600)
  so the LLM API key it holds isn't world/group-readable.
- OVH-064: ``.env.example`` must document ``TOPIC_WATCH_SECURE_COOKIES`` (commented)
  so remote deployers following SECURITY.md find it.
"""

from pathlib import Path

import yaml

_ROOT = Path(__file__).resolve().parent.parent


def test_install_sh_chmods_env_file() -> None:
    install = (_ROOT / "scripts" / "install.sh").read_text()
    assert 'chmod 600 "${ENV_FILE}"' in install


def test_env_example_documents_secure_cookies() -> None:
    env_example = (_ROOT / ".env.example").read_text()
    assert "# TOPIC_WATCH_SECURE_COOKIES=true" in env_example


def test_config_example_ships_no_live_notification_urls() -> None:
    """The shipped example is auto-copied to data/config.yml on first run; a live
    placeholder URL (e.g. ntfy://your-topic-name) would deliver to a real public
    target. It must ship with an empty urls list so nothing is sent until the
    user opts in (example-URL leak guard)."""
    data = yaml.safe_load((_ROOT / "config.example.yml").read_text())
    assert data["notifications"]["urls"] == []


def test_lock_targets_declare_their_toolchain() -> None:
    """``make lock`` must be runnable, or say exactly how to make it runnable.

    The lock targets call ``pip-compile``, which ships with pip-tools — but
    pip-tools was declared nowhere, so the command SECURITY.md tells users to run
    died with ``pip-compile: command not found``. It deliberately does NOT belong
    in requirements-dev.txt: pip-tools depends on pip/setuptools/wheel, which
    would hash-pin pip itself into the file ``make dev`` and CI install.
    """
    makefile = (_ROOT / "Makefile").read_text()
    assert "lock-tools:" in makefile, "no target installs the pip-compile toolchain"
    assert "pip-tools==" in makefile, "the pip-tools version must be pinned"
    # Both lock targets must fail with guidance rather than 'command not found'.
    for target in ("lock:", "lock-upgrade:"):
        line = next(ln for ln in makefile.splitlines() if ln.startswith(target))
        assert "_require_pip_compile" in line, f"{target} is not guarded"


def test_security_md_points_at_a_real_relock_target() -> None:
    """SECURITY.md tells users to regenerate the lockfile; that path must exist."""
    security = (_ROOT / "SECURITY.md").read_text()
    makefile = (_ROOT / "Makefile").read_text()
    assert "make lock-tools" in security
    assert "lock-tools:" in makefile


def test_env_example_has_no_uncommented_llm_key() -> None:
    """``.env`` is interpolation-only — Compose never injects it into the
    container — so a live ``TOPIC_WATCH_LLM__*`` line in .env.example is a false
    promise. The LLM lines must stay commented (set the key via the wizard)."""
    for line in (_ROOT / ".env.example").read_text().splitlines():
        stripped = line.strip()
        assert not stripped.startswith("TOPIC_WATCH_LLM__"), f"uncommented LLM env line: {line!r}"


def test_entrypoint_refuses_root_and_system_chown_targets() -> None:
    """AUG-007: a relocated TOPIC_WATCH_DB_PATH (e.g. ``/topic_watch.db`` or
    ``../topic_watch.db``) must never resolve to a directory the entrypoint
    will recursively chown. The guard must canonicalize the resolved
    directory and refuse the filesystem root and other broad system dirs."""
    entrypoint = (_ROOT / "docker-entrypoint.sh").read_text()
    assert "canon_dir" in entrypoint, "db_dir must be canonicalized before use"
    assert "is_forbidden_chown_target" in entrypoint
    # The guard must actually run before any chown of db_dir happens.
    guard_pos = entrypoint.index("is_forbidden_chown_target")
    chown_pos = entrypoint.index('chown_path_if_needed "$db_dir"')
    assert guard_pos < chown_pos, "the forbidden-target guard must precede the db_dir chown"
    # Must reject "/" itself, not just its named children.
    forbidden_block = entrypoint[entrypoint.index("is_forbidden_chown_target()") :][:400]
    assert '""|/) return 0' in forbidden_block


def test_entrypoint_never_recursively_chowns_the_configured_db_dir() -> None:
    """AUG-007's core fix: an arbitrary configured database directory must be
    touched with plain (non-recursive) ``chown``, never ``chown -R`` — only
    the fixed, bounded ``/app/data`` (and its own ``backups`` subdirectory)
    may be recursed."""
    entrypoint = (_ROOT / "docker-entrypoint.sh").read_text()
    assert 'chown -R "$PUID:$PGID" "$db_dir"' not in entrypoint
    assert 'chown_path_if_needed "$db_dir"' in entrypoint, "db_dir itself must use the non-recursive helper"


def test_entrypoint_repairs_existing_required_files_not_just_the_directory() -> None:
    """AUG-057: ownership repair must not skip existing config/database files
    just because the containing directory's owner already matches — a
    restored/copied-in file can still be root-owned."""
    entrypoint = (_ROOT / "docker-entrypoint.sh").read_text()
    assert "chown_db_files" in entrypoint
    assert 'chown_path_if_needed "$DATA_DIR/config.yml"' in entrypoint
    # The per-file repair must run unconditionally, not nested inside the
    # directory-level "ownership already matches" skip branch.
    dir_check_pos = entrypoint.index('if [ "$dir_uid" != "$PUID" ]')
    file_repair_pos = entrypoint.index('chown_path_if_needed "$DATA_DIR/config.yml"')
    assert file_repair_pos > dir_check_pos


def test_compose_files_pass_through_secure_cookies() -> None:
    """AUG-058: the documented ``TOPIC_WATCH_SECURE_COOKIES`` .env switch must
    actually reach the container — Compose's .env only fills ${...}
    placeholders, it does not become the container environment on its own."""
    for compose_file in ("docker-compose.yml", "docker-compose.prod.yml"):
        text = (_ROOT / compose_file).read_text()
        assert "TOPIC_WATCH_SECURE_COOKIES" in text, f"{compose_file} does not pass through TOPIC_WATCH_SECURE_COOKIES"


def test_install_and_update_scripts_exit_nonzero_on_failed_health_check() -> None:
    """AUG-059: install.sh, install.ps1, and update.sh must fail loudly (and
    stop before any success epilogue) when the post-start health check never
    passes, instead of warning and then reporting success anyway."""
    install_sh = (_ROOT / "scripts" / "install.sh").read_text()
    install_ps1 = (_ROOT / "scripts" / "install.ps1").read_text()
    update_sh = (_ROOT / "scripts" / "update.sh").read_text()

    # install.sh: the unhealthy branch must exit 1 before the "running!" epilogue.
    unhealthy_pos = install_sh.index('if [ "$HEALTHY" != "1" ]; then')
    exit_pos = install_sh.index("exit 1", unhealthy_pos)
    epilogue_pos = install_sh.index("Topic Watch is running!")
    assert unhealthy_pos < exit_pos < epilogue_pos

    # install.ps1: same shape.
    unhealthy_pos = install_ps1.index("if (-not $healthy) {")
    exit_pos = install_ps1.index("exit 1", unhealthy_pos)
    epilogue_pos = install_ps1.index('"Topic Watch is running!"')
    assert unhealthy_pos < exit_pos < epilogue_pos

    # update.sh: the failure branch must exit 1, not fall off the end successfully.
    failure_pos = update_sh.index('error "Health check failed after update!"')
    exit_pos = update_sh.index("exit 1", failure_pos)
    assert exit_pos > failure_pos


def test_update_sh_reads_port_from_persisted_env_not_process_env_only() -> None:
    """AUG-060 remnant: update.sh must fall back to the port persisted in the
    install's .env (the same source docker-compose.yml reads), not only the
    invoking process's environment, or a custom-port install gets health
    checked on the wrong port after the process env is gone."""
    update_sh = (_ROOT / "scripts" / "update.sh").read_text()
    assert "read_env TOPIC_WATCH_PORT" in update_sh
    assert 'PORT="${TOPIC_WATCH_PORT:-$(read_env TOPIC_WATCH_PORT "$ENV_FILE")}"' in update_sh


def test_installers_fetch_the_ollama_override_example() -> None:
    """AUG-073 remnant: the README's documented
    ``cp docker-compose.override.example.yml docker-compose.override.yml``
    step needs the example file present outside a source checkout too, so
    the scripted installers must fetch it (best-effort; it's optional)."""
    install_sh = (_ROOT / "scripts" / "install.sh").read_text()
    install_ps1 = (_ROOT / "scripts" / "install.ps1").read_text()
    assert "docker-compose.override.example.yml" in install_sh
    assert "docker-compose.override.example.yml" in install_ps1


def test_image_is_pinned_and_recorded_across_install_and_update() -> None:
    """TW-AUD-032: the production image reference must be parameterized so a
    verified-healthy digest can be pinned into .env at install/update time,
    and restarts (reboot, systemd, a later `docker compose up`) then reuse
    that recorded digest instead of silently re-resolving movable "latest"."""
    compose_prod = (_ROOT / "docker-compose.prod.yml").read_text()
    assert "${TOPIC_WATCH_IMAGE:-ghcr.io/0xzerolight/topic_watch:latest}" in compose_prod

    install_sh = (_ROOT / "scripts" / "install.sh").read_text()
    assert "docker inspect --format '{{index .RepoDigests 0}}'" in install_sh
    assert 'upsert_env "TOPIC_WATCH_IMAGE"' in install_sh

    update_sh = (_ROOT / "scripts" / "update.sh").read_text()
    assert "docker inspect --format '{{index .RepoDigests 0}}'" in update_sh
    assert 'upsert_env "TOPIC_WATCH_IMAGE" "$NEW_DIGEST"' in update_sh
    # Rollback must restore the previously recorded digest, not just retry
    # the now-current (possibly broken) image reference.
    assert 'upsert_env "TOPIC_WATCH_IMAGE" "$PREV_IMAGE"' in update_sh
