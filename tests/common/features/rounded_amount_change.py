from dataclasses import dataclass
from typing import Type, Tuple

import pandas as pd

from feature_forger.domain.entities.entity_model import EntityModel
from feature_forger.domain.entities.feature import Feature
from tests.common.entity_models.transaction import Transaction
from tests.common.features.transaction_amount_change import \
    TransactionAmountChange


@dataclass(frozen=True)
class TransactionRoundedAmountChange(Feature):
    col_name: str = "transaction_rounded_amount_change"
    description: str = "rounded difference between withdrawal amount and desposit " \
                  "amount"
    entity_model: Type[EntityModel] = Transaction
    dependencies: Tuple[Feature] = (TransactionAmountChange(),)

    def row_compute_fn(self, row: pd.Series):
        row[self.col_name] = row[TransactionAmountChange.col_name] / 1000
        return row

    def table_compute_fn(self, data: pd.DataFrame):
        return self.row_compute_fn(data)

