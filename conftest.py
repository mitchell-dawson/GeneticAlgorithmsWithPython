
import shutil
from pathlib import Path
from typing import Generator

import pytest


@pytest.fixture
def data_fixtures_folder():
    return Path(__file__).parent.absolute() / "data_fixtures"

@pytest.fixture(name="temp_folder")
def fixture_temp_folder() -> Generator[Path, None, None]:
    """Create a temporary folder, and yield the path to it. The folder is
    deleted after
    Yields
    ------
    Path
        Path to the temporary folder
    """
    test_folder = Path(__file__).parent.absolute()
    temporary_test_folder = test_folder / "temp"
    temporary_test_folder.mkdir(exist_ok=True)
    yield temporary_test_folder
    shutil.rmtree(temporary_test_folder)
