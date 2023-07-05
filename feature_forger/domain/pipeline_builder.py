import itertools
from copy import deepcopy
from itertools import groupby
from typing import List, Tuple, Union

import networkx as nx
import pandas as pd
from prefect import flow, task

from feature_forger.domain.composite_task import CompositeTask
from feature_forger.domain.dag_builder import DAGBuilder
from feature_forger.domain.dataset_model_builder import DatasetModelBuilder
from feature_forger.domain.entities.dataset import Dataset
from feature_forger.domain.entities.feature import Feature
from feature_forger.domain.entities.pipeline import Pipeline
from feature_forger.domain.feature_validator import FeatureValidator


class PipelineBuilder:

    def __init__(self,
                 dataset_model_builder: DatasetModelBuilder,
                 dag_builder: DAGBuilder,
                 feature_validator: FeatureValidator):
        self._dataset_model_builder = dataset_model_builder
        self._dag_builder = dag_builder
        self._feature_validator = feature_validator

    def build(self,
              dataset: Dataset,
              features: List[Feature],
              concurrent_paths: bool) -> List[Pipeline]:

        entity_feature_map = {entity: list(features)
                              for entity, feature_group
                              in groupby(features,
                                         lambda f: f.entity_model)}
        unsupported_entities = set(entity_feature_map.keys()).difference(
            dataset.supported_entity_models)
        if unsupported_entities:
            raise AttributeError('Features depend on entities not '
                                 f'supported by this dataset ({unsupported_entities = })')
        pipelines = []
        for entity in entity_feature_map.keys():
            if concurrent_paths:
                pipeline = self._build_concurrent_paths_flow(
                    dataset=dataset,
                    entity=entity,
                    entity_feature_map=entity_feature_map
                )
                pipelines.append(pipeline)
            else:
                pipeline = self._build_single_path_flow(
                    dataset=dataset,
                    entity=entity,
                    entity_feature_map=entity_feature_map
                )
                pipelines.append(pipeline)
        return pipelines

    def _build_single_path_flow(self, dataset, entity, entity_feature_map):
        graph = self._dag_builder.build_simple_graph(
            features=entity_feature_map[entity],
            entity=entity
        )
        ordered_features = [f for f in nx.topological_sort(graph) if
                            isinstance(f, Feature)]

        func_map = self._build_feature_func_map(ordered_features)
        tasks = self._create_composite_tasks(func_map)

        return Pipeline(
            dataset=dataset,
            entity=entity,
            flow=self._build_pipeline(
                ordered_tasks=tasks
            ),
            graph=self._dag_builder.build_pydot(
                graph=graph,
                label_map={str(n): n.name
                           for n in
                           nx.nodes(graph)}),
            requested_features=[f.col_name for f in
                                entity_feature_map[entity]],
            all_features=[f.col_name for f in ordered_features]

        )

    def _create_composite_tasks(self, func_map):
        tasks = []
        for key, grouped in groupby(func_map, key=lambda x: x[0]):
            fs = [g[1] for g in grouped]
            if key == 'row':
                tasks.append(CompositeTask(
                    type=key, funcs=[
                        self._feature_validator.add_runtime_validation(
                            f.row_compute_fn) for f in fs],
                    features=[f.col_name for f in fs]))
            else:
                tasks += [
                    CompositeTask(
                        type=key,
                        funcs=[
                            self._feature_validator.add_runtime_validation(
                                f.table_compute_fn)
                        ], features=[f.col_name]
                    )
                    for f in fs
                ]
        return tasks

    def _build_feature_func_map(self, ordered_features):
        func_map = []
        for feature in ordered_features:
            if not isinstance(feature, Feature):
                continue
            if feature.table_compute_fn.__func__ != \
                    Feature.table_compute_fn:
                func_map.append(('table', feature))
            else:
                func_map.append(('row', feature))
        return func_map

    def _build_concurrent_paths_flow(self, dataset, entity, entity_feature_map
                                     ):
        flows = []
        graph = self._dag_builder.build_simple_graph(
            features=entity_feature_map[entity],
            entity=entity)
        group_graph = self._dag_builder.build_grouped_graph(deepcopy(graph))
        all_features = []
        for group in nx.topological_sort(group_graph):
            predecessors = list(group_graph.in_edges(group))
            predecessor_nodes = tuple(p[0] for p in predecessors)
            group = (group,) if not isinstance(group, tuple) else group
            func_map = self._build_feature_func_map(group)
            grouped_ordered = self._create_composite_tasks(func_map)

            if grouped_ordered:
                flows.append((predecessor_nodes,
                              self._build_group_pipeline(group,
                                                         grouped_ordered)))
                all_features.append([t.features for t in grouped_ordered])

        pipeline = Pipeline(
            dataset=dataset,
            entity=entity,
            flow=self._build_main_pipeline(flows, entity),
            graph=self._dag_builder.build_pydot(
                graph=graph,
                label_map={
                    str(node): self._get_name_from_group_node(
                        node)
                    for node in nx.nodes(graph)}
            ),
            requested_features=[f.col_name for f in
                                entity_feature_map[entity]],
            all_features=[f for f in
                          self._flatten_feature_list(all_features)]
        )
        return pipeline

    def _flatten_feature_list(self, lst):
        return [item for sublist in lst for item in
                (self._flatten_feature_list(sublist) if isinstance(sublist,
                                                                   list) else [
                    sublist])]

    def _build_main_pipeline(self, flows, entity):

        def pipeline(x):
            dfs = {entity.name: x}
            original_cols = x.columns
            for predecessor_nodes, flow in flows:
                kwargs = {}
                wait_for = [self._get_name_from_group_node(p)
                            for p in predecessor_nodes] or None
                if wait_for:
                    kwargs['wait_for'] = [dfs[n] for n in wait_for]

                dfs[flow.name] = flow(x, **kwargs)
            dfs[entity.name] = dfs[entity.name].loc[:, original_cols]
            return self._concat_task(dfs=dfs.values(),
                                     wait_for=[dfs[k] for k in dfs.keys()])

        return flow(name=f"forging-{entity.name}-features-concurrent")(
            pipeline)

    @staticmethod
    @task(name='combine')
    def _concat_task(dfs):
        return pd.concat(dfs, axis=1)

    def _get_name_from_group_node(self,
                                  group: Union[Feature, Tuple[Feature]]):
        if not isinstance(group, tuple):
            return group.name
        return ' --} '.join(f.name for f in group)

    def _build_group_pipeline(self,
                              node: Tuple[Feature],
                              ordered_tasks: List[CompositeTask]):
        def pipeline(x):
            for t in ordered_tasks:
                x = t.prefect_task(x)
            return x[
                list(itertools.chain(*[t.features for t in ordered_tasks]))]

        return flow(name=self._get_name_from_group_node(node))(pipeline)

    def _build_pipeline(self, ordered_tasks: List[CompositeTask]):

        def pipeline(x):
            for t in ordered_tasks:
                x = t.task(x)
            return x

        return pipeline

    def _build_prefect_pipeline(self,
                                name: str,
                                ordered_tasks: List[CompositeTask]):

        def pipeline(x):
            for t in ordered_tasks:
                x = t.prefect_task(x)
            return x

        return flow(name=name)(pipeline)
