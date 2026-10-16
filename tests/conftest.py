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


@pytest.fixture(autouse=True)
def _reset_offline() -> None:
    """`--offline` sets a module-level switch; reset it so one CLI test cannot leak it into the next."""
    from dist_s1_prod_check import network

    yield
    network.set_offline(False)
