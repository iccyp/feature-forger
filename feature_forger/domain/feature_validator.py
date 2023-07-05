from functools import wraps
from typing import Union

import pandas as pd

from feature_forger.domain.entities.feature import Feature


class FeatureValidator:

    def add_runtime_validation(self, compute_fn: Union[Feature.row_compute_fn,
                             Feature.table_compute_fn]):
        return self._validate_feature_compute_fn(compute_fn)

    def _validate_feature_compute_fn(self, func):
        inst = func.__self__

        @wraps(func)
        def wrapper(data: Union[pd.DataFrame, pd.Series]) -> Union[
            pd.DataFrame, pd.Series]:
            if inst.col_name in data:
                raise KeyError(
                    f"Feature {inst.col_name} already exists in the "
                    f"dataset. Rename this column or the feature col_name")
            result = func(data)
            if not inst.col_name in result:
                raise KeyError(f"Feature {inst.col_name} has not created "
                               f"the feature in the dataset.")
            return result

        return wrapper
