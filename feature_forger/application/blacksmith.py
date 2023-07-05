from functools import cached_property
from typing import List, Type

import pandas as pd

from feature_forger.application.container import Container
from feature_forger.domain.entities.dataset import Dataset
from feature_forger.domain.entities.feature import Feature
from feature_forger.domain.entities.pipeline import Pipeline
from feature_forger.domain.pipeline_builder import PipelineBuilder
from feature_forger.domain.pipeline_runner import PipelineRunner
from feature_forger.application.models.recipe import Recipe


class Blacksmith:

    def __init__(self):
        self._container = Container()
        self._container.wire([__name__])
        self._container.init_resources()

    @cached_property
    def _pipeline_builder(self) -> PipelineBuilder:
        return self._container.domain.pipeline_builder()

    @cached_property
    def _pipeline_runner(self) -> PipelineRunner:
        return self._container.domain.pipeline_runner()

    def build_recipes(self,
                      dataset: Dataset,
                      features: List[Feature],
                      concurrent_paths: bool = False) -> List[Recipe]:
        pipelines = self._pipeline_builder.build(
            dataset=dataset,
            features=features,
            concurrent_paths=concurrent_paths
        )
        return [Recipe(dataset=pipeline.dataset,
                       entity=pipeline.entity,
                       flow=pipeline.flow,
                       graph=pipeline.graph,
                       requested_features=pipeline.requested_features,
                       all_features=pipeline.all_features,
                       ) for pipeline in pipelines]

    def forge(self, recipe: Recipe, copy: bool = True) -> pd.DataFrame:
        return self._pipeline_runner.run(pipeline=Pipeline(
            dataset=recipe.dataset,
            entity=recipe.entity,
            flow=recipe.flow,
            graph=recipe.graph,
            requested_features=recipe.requested_features,
            all_features=recipe.all_features
        ), copy=copy)



