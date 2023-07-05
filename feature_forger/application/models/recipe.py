import io
from typing import Type, List, Callable

import pandas as pd
from PIL import Image
from pydantic import BaseModel
from pydot import Dot

from feature_forger.domain.entities.dataset import Dataset
from feature_forger.domain.entities.entity_model import EntityModel


class Recipe(BaseModel):
    dataset: Dataset
    entity: Type[EntityModel]
    flow: Callable[[pd.DataFrame], pd.DataFrame]
    graph: Dot
    requested_features: List[str]
    all_features: List[str]

    class Config:
        arbitrary_types_allowed = True
        frozen = True

    def __str__(self):
        return self.__repr__()

    def __repr_args__(self):
        return [(a, v) if a not in ['dataset']
                else (a, self.dataset.__class__.__name__)
                for a, v in super().__repr_args__()]

    def view(self):
        image = Image.open(io.BytesIO(self.graph.create_png()))
        image.show()
