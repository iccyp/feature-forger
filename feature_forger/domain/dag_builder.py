import itertools
from typing import List, Type, Dict

import networkx as nx
from networkx import DiGraph
from pydot import Dot

from feature_forger.domain.entities.entity_model import EntityModel
from feature_forger.domain.entities.feature import Feature


class DAGBuilder:

    def build_simple_graph(self,
                           features: List[Feature],
                           entity: Type[EntityModel]):
        graph = nx.DiGraph()
        for feature in features:
            graph.add_edges_from([(entity, feature)])
            graph = self._add_dependencies(graph, entity, feature)

        nx.is_directed_acyclic_graph(graph)
        graph = nx.transitive_reduction(graph)
        return graph

    def _follow_path_until_split(self, graph, node, inline_edges=None):
        inline_edges = inline_edges or []

        out_edges = graph.out_edges(node)
        in_edges = graph.in_edges(node)

        number_of_out_edges = len(out_edges)
        number_of_in_edges = len(in_edges)

        if number_of_in_edges < 2:
            inline_edges.append(node)
        else:
            return inline_edges

        if number_of_out_edges != 1:
            return inline_edges

        out_node = list(out_edges)[0][1]
        return self._follow_path_until_split(graph, out_node, inline_edges=inline_edges)

    def build_grouped_graph(self,
                            single_path_graph: DiGraph):
        graph = single_path_graph

        inline_nodes = []
        for node in graph.nodes:
            sequenced_nodes = self._follow_path_until_split(graph, node)
            if len(sequenced_nodes) > 1:
                inline_nodes.append(tuple(sequenced_nodes))

        inline_nodes = sorted(inline_nodes, key=lambda x: -len(x))

        supernodes = []
        for group in inline_nodes:
            group_len = len(group)
            group_is_sublist = False
            for supernode in supernodes:
                supernode_len = len(supernode)
                if supernode == group:
                    group_is_sublist = True
                for i in range(supernode_len - group_len + 1):
                    if supernode[i:i+group_len] == group:
                        group_is_sublist = True
            if not group_is_sublist:
                supernodes.append(group)

        for supernode in supernodes:
            out_nodes = [e[1] for e in graph.out_edges(supernode[-1])]
            in_nodes = [e[0] for e in graph.in_edges(supernode[0])]

            for node in supernode:
                graph.remove_node(node)

            graph.add_node(supernode)
            graph.add_edges_from([(i, supernode) for i in in_nodes])
            graph.add_edges_from([(supernode, o) for o in out_nodes])

        return nx.transitive_reduction(graph)

    def build_pydot(self,
                    graph: DiGraph,
                    label_map: Dict[str, str]) -> Dot:
        dot_graph = nx.nx_pydot.to_pydot(graph)
        for node in dot_graph.get_nodes():
            node.set_label(label_map[node.get_name().replace('"', '')])
        return dot_graph

    def _add_dependencies(self, graph, entity, feature, dependencies=None):
        dependencies = dependencies or []
        for dep in feature.dependencies:
            graph.add_edges_from([(entity, dep), (dep, feature)])
            self._add_dependencies(graph, entity, dep,
                                   dependencies=dependencies)
        return graph


