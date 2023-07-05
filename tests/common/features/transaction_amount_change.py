from dataclasses import dataclass
from typing import Type, Tuple

import numpy as np
import pandas as pd

from feature_forger.domain.entities.entity_model import EntityModel
from feature_forger.domain.entities.feature import Feature
from tests.common.entity_models.transaction import \
    Transaction


@dataclass(frozen=True)
class TransactionAmountChange(Feature):
    col_name: str = "transaction_amount_change"
    description: str = "difference between witdrawal amount and deposit amount"
    entity_model: Type[EntityModel] = Transaction
    dependencies: Tuple[Feature] = tuple()

    def row_compute_fn(self, row: pd.Series):
        original_withdrawal_amt = row[Transaction['withdrawal_amount']]
        original_deposit_amount = row[Transaction['deposit_amount']]
        withdrawal_amount = 0 if np.isnan(original_withdrawal_amt) \
            else original_withdrawal_amt
        deposit_amount = 0 if np.isnan(original_deposit_amount) \
            else original_deposit_amount
        row[self.col_name] = withdrawal_amount - deposit_amount
        return row

    def table_compute_fn(self, data: pd.DataFrame) -> pd.Series:
        data[self.col_name] = data[Transaction['withdrawal_amount']].fillna(0) - data[Transaction['deposit_amount']].fillna(0)
        return data