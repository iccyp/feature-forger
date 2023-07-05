from dataclasses import dataclass
from typing import Type

from feature_forger.domain.entities.dataset import Dataset
from feature_forger.domain.entities.entity_model import EntityModel
from tests.common.entity_models.account import Account
from tests.common.entity_models.transaction import \
    Transaction


@dataclass(frozen=True)
class BankDataset(Dataset):
    supported_entity_models = (Transaction, Account)

    def map_rows_to_entity(self, entity: Type[EntityModel]):
        if entity == Transaction:
            return self.data
        if entity == Account:
            return self.data.groupby(Account['account_no'])

