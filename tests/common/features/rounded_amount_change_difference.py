from dataclasses import dataclass
from typing import Type, Tuple

import pandas as pd

from feature_forger.domain.entities.entity_model import EntityModel
from feature_forger.domain.entities.feature import Feature
from tests.common.entity_models.transaction import Transaction
from tests.common.features.transaction_month import TransactionMonth
from tests.common.features.rounded_amount_change import \
    TransactionRoundedAmountChange
from tests.common.features.transaction_amount_change import \
    TransactionAmountChange

@dataclass(frozen=True)
class TransactionRoundedAmountChangeDiff(Feature):
    col_name: str = "transaction_rounded_amount_change_diff"
    description: str = "difference of the rounded difference between the withdrawal" \
                  " amount and the deposit amount"
    entity_model: Type[EntityModel] = Transaction
    dependencies: Tuple[Feature] = (
        TransactionRoundedAmountChange(),
        TransactionAmountChange(),
        TransactionMonth()
    )

    def row_compute_fn(self, row: pd.Series):
        row[self.col_name] = row[TransactionRoundedAmountChange.col_name] - row[TransactionAmountChange.col_name] + row[TransactionMonth.col_name]
        return row

    def table_compute_fn(self, data: pd.DataFrame):
        return self.row_compute_fn(data)
