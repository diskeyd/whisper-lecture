"""Machine-specific settings come from the environment or local/config.toml, never from the code."""

import pytest
import whisper_config as cfg


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    """No real config file, no inherited variables."""
    for name in cfg.KEYS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(cfg, "CONFIG_FILE", tmp_path / "config.toml")
    return tmp_path / "config.toml"


def test_nothing_set_means_none(isolated):
    assert cfg.get("WHISPER_DRIVE_FOLDER_ID") is None


def test_environment_variable_is_read(isolated, monkeypatch):
    monkeypatch.setenv("WHISPER_DRIVE_FOLDER_ID", " abc123 ")
    assert cfg.get("WHISPER_DRIVE_FOLDER_ID") == "abc123"


def test_file_is_the_fallback_and_the_environment_wins(isolated, monkeypatch):
    isolated.write_text('wiki_vault = "~/notes"\ndrive_folder_id = "from-file"\n', encoding="utf-8")
    assert cfg.get("WHISPER_DRIVE_FOLDER_ID") == "from-file"
    monkeypatch.setenv("WHISPER_DRIVE_FOLDER_ID", "from-env")
    assert cfg.get("WHISPER_DRIVE_FOLDER_ID") == "from-env"
    assert cfg.path("WHISPER_WIKI_VAULT").name == "notes"
    assert "~" not in str(cfg.path("WHISPER_WIKI_VAULT"))


def test_broken_file_stops_the_run_with_its_path(isolated):
    isolated.write_text("vault = \n", encoding="utf-8")
    with pytest.raises(cfg.ConfigError, match="config.toml"):
        cfg.get("WHISPER_VAULT")


def test_unset_path_can_be_extended_but_not_used(isolated):
    root = cfg.path("WHISPER_WIKI_VAULT")
    child = root / "10. Raw Sources" / "course"
    with pytest.raises(cfg.ConfigError, match="WHISPER_WIKI_VAULT"):
        child.iterdir()
    with pytest.raises(cfg.ConfigError, match="WHISPER_WIKI_VAULT"):
        str(child)


def test_a_set_path_is_a_plain_path(isolated, monkeypatch, tmp_path):
    monkeypatch.setenv("WHISPER_VAULT", str(tmp_path))
    assert cfg.path("WHISPER_VAULT") == tmp_path


def test_no_values_are_hard_coded_in_the_repo():
    """The placeholders in config.example.toml are the only vault or Drive values the repo carries."""
    example = (cfg.ROOT / "config.example.toml").read_text(encoding="utf-8")
    for key in cfg.KEYS.values():
        assert f"{key} =" in example
    assert "/Users/" not in example


def test_an_uploading_run_without_the_vault_stops_before_transcribing(
    isolated, monkeypatch, tmp_path, capsys
):
    import sys
    import types

    sys.modules.setdefault("mlx_whisper", types.ModuleType("mlx_whisper"))
    sys.path.insert(0, str(cfg.ROOT))
    import transcribe_whisper as tw

    audio = tmp_path / "talk.m4a"
    audio.write_bytes(b"")
    monkeypatch.setattr(tw, "SUBJECT_DIR", cfg.UnsetPath("WHISPER_VAULT"))
    monkeypatch.setattr(sys, "argv", ["transcribe_whisper.py", str(audio)])
    with pytest.raises(SystemExit) as stop:
        tw.main()
    assert stop.value.code == 1
    assert "WHISPER_VAULT" in capsys.readouterr().out


def test_a_missing_drive_folder_fails_the_upload_and_names_the_variable(monkeypatch, tmp_path):
    import sys
    import types

    sys.modules.setdefault("mlx_whisper", types.ModuleType("mlx_whisper"))
    sys.path.insert(0, str(cfg.ROOT))
    import transcribe_whisper as tw

    monkeypatch.setattr(tw, "DRIVE_FOLDER_ID", None)
    file_id, note = tw.upload_to_drive(tmp_path / "x.m4a")
    assert file_id is None and "WHISPER_DRIVE_FOLDER_ID" in note
