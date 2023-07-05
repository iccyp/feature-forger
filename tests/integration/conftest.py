from pathlib import Path

import pandas as pd
import pytest


@pytest.fixture(scope='session')
def data() -> pd.DataFrame:
    return pd.read_excel(Path(__file__).parents[1].joinpath(
        'common', 'data', 'bank.xlsx').as_posix(), nrows=5_000)
