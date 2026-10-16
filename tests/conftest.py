from pathlib import Path

import pytest
from click.testing import CliRunner


@pytest.fixture
def tmp_data_dir(tmp_path: Path) -> Path:
    data_dir = tmp_path / 'data'
    data_dir.mkdir()
    return data_dir


@pytest.fixture
def cli_runner() -> CliRunner:
    return CliRunner()
