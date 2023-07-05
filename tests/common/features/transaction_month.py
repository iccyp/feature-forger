from typing import Type, Tuple, Optional

import pandas as pd
from pydantic import Field
from pydantic.dataclasses import dataclass

from feature_forger.domain.entities.entity_model import EntityModel
from feature_forger.domain.entities.feature import Feature
from tests.common.entity_models.transaction import \
    Transaction


@dataclass(frozen=True)
class TransactionMonth(Feature):
    col_name: str = "value_month"
    description: str = "month of the value date for the transaction"
    entity_model: Type[EntityModel] = Transaction
    dependencies: Optional[Tuple[Feature]] = Field(default_factory=tuple)

    def row_compute_fn(self, data: pd.Series):
        data[self.col_name] = data[Transaction['value_date']].month
        return data

    def table_compute_fn(self, data: pd.DataFrame):
        data[self.col_name] = data[Transaction['value_date']].dt.month
        return data
