from dataclasses import dataclass
from functools import reduce, cached_property
from typing import List, Callable, Union

import pandas as pd
from prefect import task

from feature_forger.domain.entities.feature import Feature


@dataclass
class CompositeTask:
    type: str
    funcs: List[Union[Feature.row_compute_fn, Feature.table_compute_fn]]
    features: List[str]

    def _build_elementwise_function(self):
        def elementwise_fn(x):
            x = x.apply(self.fn, axis=1)
            return x

        return elementwise_fn

    def _build_vectorised_function(self):
        def vectorized_fn(x):
            x = self.fn(x)
            return x

        return vectorized_fn

    @cached_property
    def prefect_task(self):
        return task(name=self.name)(self.task)

    @cached_property
    def task(self):
        if self.type == 'row':
            return self._build_elementwise_function()
        return self._build_vectorised_function()

    @cached_property
    def fn(self) -> Callable:
        if len(self.funcs) == 1:
            return self.funcs[0]

        def compose(f: Callable[[pd.Series], pd.Series],
                    g: Callable[[pd.Series], pd.Series]):
            return lambda x: g(f(x))

        return reduce(compose, self.funcs, lambda x: x)

    @cached_property
    def name(self) -> str:
        return ','.join(self.features)
