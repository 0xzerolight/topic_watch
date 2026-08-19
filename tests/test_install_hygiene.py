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
