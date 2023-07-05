from typing import List

from pydantic import BaseModel
from pydantic.fields import ModelField


class EntityModel(BaseModel):

    @classmethod
    def __class_getitem__(cls, item) -> str:
        field: ModelField = cls.__fields__.get(item)
        return field.alias

    @classmethod
    def columns(cls) -> List[str]:
        return [f.alias for f in cls.__fields__.values()]

    @classmethod
    @property
    def name(cls) -> str:
        return cls.__name__