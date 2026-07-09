"""Tests for shared config loading (docforge.config), used by both cli.py and mcp_server.py."""

from docforge.config import resolve_settings


def test_resolve_settings_defaults_when_nothing_set(monkeypatch) -> None:
    for key in ("DOCFORGE_DB", "QDRANT_URL", "QDRANT_PATH", "QDRANT_API_KEY", "DOCFORGE_EMBED_MODEL"):
        monkeypatch.delenv(key, raising=False)

    settings = resolve_settings()

    assert settings["db_path"] == "docforge.db"
    assert settings["qdrant_url"] == "http://localhost:6333"
    assert settings["qdrant_path"] is None
    assert settings["qdrant_api_key"] is None
    assert settings["qdrant_timeout"] == 60.0
    assert settings["embed_model"] == "BAAI/bge-small-en-v1.5"


def test_resolve_settings_env_overrides_default(monkeypatch) -> None:
    monkeypatch.setenv("DOCFORGE_DB", "SqlDB/docforge.db")
    monkeypatch.setenv("QDRANT_PATH", "./vectors")
    monkeypatch.setenv("DOCFORGE_EMBED_MODEL", "BAAI/bge-base-en-v1.5")

    settings = resolve_settings()

    assert settings["db_path"] == "SqlDB/docforge.db"
    assert settings["qdrant_path"] == "./vectors"
    assert settings["embed_model"] == "BAAI/bge-base-en-v1.5"


def test_resolve_settings_explicit_arg_wins_over_env(monkeypatch) -> None:
    monkeypatch.setenv("DOCFORGE_DB", "SqlDB/docforge.db")
    monkeypatch.setenv("DOCFORGE_EMBED_MODEL", "BAAI/bge-base-en-v1.5")

    settings = resolve_settings(db_path="explicit.db", embed_model="BAAI/bge-large-en-v1.5")

    assert settings["db_path"] == "explicit.db"
    assert settings["embed_model"] == "BAAI/bge-large-en-v1.5"
