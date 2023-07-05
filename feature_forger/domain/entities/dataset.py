from dataclasses import dataclass
from typing import Type, Protocol, Tuple, runtime_checkable, ClassVar

import pandas as pd

from feature_forger.domain.entities.entity_model import EntityModel


@dataclass(frozen=True)
class Dataset:
    supported_entity_models: ClassVar[Tuple[Type[EntityModel]]]
    data: pd.DataFrame

    def map_rows_to_entity(self, entity: Type[EntityModel]):
        """Maps the dataframe such that rows to entities is 1-to-1."""
        ...
