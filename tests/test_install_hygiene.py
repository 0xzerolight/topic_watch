"""Static checks for install-script / env-template secret hygiene.

These guard the non-Python deliverables of Task 2.3:
- OVH-063: ``scripts/install.sh`` must lock down the generated ``.env`` (chmod 600)
  so the LLM API key it holds isn't world/group-readable.
- OVH-064: ``.env.example`` must document ``TOPIC_WATCH_SECURE_COOKIES`` (commented)
  so remote deployers following SECURITY.md find it.
"""

from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent


def test_install_sh_chmods_env_file() -> None:
    install = (_ROOT / "scripts" / "install.sh").read_text()
    assert 'chmod 600 "${ENV_FILE}"' in install


def test_env_example_documents_secure_cookies() -> None:
    env_example = (_ROOT / ".env.example").read_text()
    assert "# TOPIC_WATCH_SECURE_COOKIES=true" in env_example
