"""Runtime initialization must be explicit and preserve imported state references."""

from unittest.mock import patch

from digit.core import base
from digit.core.settings import GeneratorConfig, Settings


def test_initialize_runtime_preserves_shared_settings_and_loads_active_state(
    monkeypatch,
):
    imported_config = base.CONFIG
    settings = Settings()
    settings.generator = GeneratorConfig(id="runtime-generator")
    logging_calls = []
    monkeypatch.setattr(base, "initialize_logging", lambda: logging_calls.append(True))
    monkeypatch.setattr(base, "load_config", lambda root, logger: settings)
    with patch.dict(base.CONFIG.__dict__), patch.dict(base.CONFIG.__pydantic_private__):
        assert base.initialize_runtime() is imported_config
        assert imported_config.generator.id == "runtime-generator"
        assert logging_calls == [True]


def test_import_does_not_load_config_or_create_logs(tmp_path):
    import os
    import subprocess
    import sys

    environment = dict(os.environ, DIGIT_PROJECT_ROOT=str(tmp_path))
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from digit.core.base import CONFIG; assert CONFIG.generator is None",
        ],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 0, result.stderr
    assert list(tmp_path.iterdir()) == []
