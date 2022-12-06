
from pathlib import Path

import pytest


@pytest.fixture
def data_fixtures_folder():
    return Path(__file__).parent.absolute() / "data_fixtures"
