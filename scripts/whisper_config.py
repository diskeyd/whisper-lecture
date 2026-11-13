"""Machine-specific settings: the two vault folders and the Drive folder id.

Each value comes from an environment variable first, then from `local/config.toml`
(git-ignored; `config.example.toml` shows the format).

    WHISPER_VAULT            vault whose "30. Permanent Notes" folder holds the Subject notes
    WHISPER_WIKI_VAULT       vault whose "10. Raw Sources" folder receives the transcripts
    WHISPER_DRIVE_FOLDER_ID  Google Drive folder the recordings are uploaded to

A missing value is not an error at import time, so the measurement tools, the tests and
`--plan` work without any setup. A vault path that is not set becomes an UnsetPath: building
a longer path from it is fine, but the first real use stops the run with a message that names
the variable.
"""

import os
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONFIG_FILE = ROOT / "local" / "config.toml"

# environment variable -> key in local/config.toml
KEYS = {
    "WHISPER_VAULT": "vault",
    "WHISPER_WIKI_VAULT": "wiki_vault",
    "WHISPER_DRIVE_FOLDER_ID": "drive_folder_id",
}


class ConfigError(SystemExit):
    """A required setting is missing. Raised as SystemExit so the message ends the run."""


def _from_file():
    if not CONFIG_FILE.is_file():
        return {}
    try:
        with CONFIG_FILE.open("rb") as f:
            return tomllib.load(f)
    except (OSError, tomllib.TOMLDecodeError) as e:
        raise ConfigError(f"Error: cannot read {CONFIG_FILE}: {e}") from e


def get(name):
    """The setting's value, or None when neither the environment nor the file has it."""
    value = os.environ.get(name, "").strip()
    if value:
        return value
    value = _from_file().get(KEYS[name])
    return value.strip() if isinstance(value, str) and value.strip() else None


def missing_message(name):
    return (
        f'Error: {name} is not set. Export it, or put `{KEYS[name]} = "..."` in '
        "local/config.toml (see config.example.toml)."
    )


class UnsetPath:
    """Stands in for a path that is not configured.

    `unset / "child"` still works, so module-level constants can be built from it; anything
    else (listing, joining to a real path, converting to text) raises ConfigError.
    """

    def __init__(self, name):
        self._name = name

    def __truediv__(self, other):
        return self

    def __getattr__(self, attr):
        if attr.startswith("__"):
            raise AttributeError(attr)
        raise ConfigError(missing_message(self._name))

    def __fspath__(self):
        raise ConfigError(missing_message(self._name))

    def __str__(self):
        raise ConfigError(missing_message(self._name))

    def __repr__(self):
        return f"UnsetPath({self._name})"


def path(name):
    """The setting as a Path (`~` expanded), or an UnsetPath when it is not set."""
    value = get(name)
    return Path(value).expanduser() if value else UnsetPath(name)
