from dataclasses import dataclass
from typing import Type, Tuple

import pandas as pd

from feature_forger.domain.entities.entity_model import EntityModel
from feature_forger.domain.entities.feature import Feature
from tests.common.entity_models.transaction import Transaction
from tests.common.features.rounded_amount_change_difference import \
    TransactionRoundedAmountChangeDiff


@dataclass(frozen=True)
class TransactionRoundedAmountChangeDiffDoubled(Feature):
    col_name: str = "transaction_rounded_amount_change_diff_doubled"
    description: str = "difference of the rounded difference between the " \
                       "withdrawal" \
                       " amount and the deposit amount doubled"
    entity_model: Type[EntityModel] = Transaction
    dependencies: Tuple[Feature] = (TransactionRoundedAmountChangeDiff(),)

    def row_compute_fn(self, row: pd.Series):
        row[self.col_name] = row[
                                TransactionRoundedAmountChangeDiff.col_name] * 2
        return row

    def table_compute_fn(self, data: pd.DataFrame):
        return self.row_compute_fn(data)
