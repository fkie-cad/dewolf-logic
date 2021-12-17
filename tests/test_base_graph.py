from pytest import raises

from simplifier.world.graphs.basegraph import BaseGraph
from simplifier.world.graphs.interface import BasicEdge, BasicNode


class TestBaseGraph:
    def test_substitute_edge(self):
        """Test edge substitution."""
        n1, n2, n3 = BasicNode(1), BasicNode(2), BasicNode(3)
        graph = BaseGraph()
        graph.add_nodes_from([n1, n2, n3])
        graph.add_edge(e1 := BasicEdge(n1, n2))
        assert graph.has_edge(n1, n2)
        graph.substitute(e1, BasicEdge(n1, n3))
        assert graph.has_edge(n1, n3) and not graph.has_edge(n1, n2)

    def test_substitute_node(self):
        """Test node substitution."""
        n1, n2, n3, n4 = BasicNode(1), BasicNode(2), BasicNode(3), BasicNode(4)
        e1, e2, e3 = BasicEdge(n1, n2), BasicEdge(n3, n2), BasicEdge(n2, n4)
        graph = BaseGraph()
        graph.add_nodes_from([n1, n2, n3, n4])
        graph.add_edges_from([e1, e2, e3])
        n0 = BasicNode(0)
        graph.substitute(n2, n0)
        assert graph.has_edge(n1, n0) and graph.has_edge(n3, n0) and graph.has_edge(n0, n4)
        assert n2 not in graph

    def test_substitute_incompatible(self):
        """Test the substitute helper method with incompatible types."""
        n1, n2, n3 = BasicNode(1), BasicNode(2), BasicNode(3)
        graph = BaseGraph()
        graph.add_nodes_from([n1, n2, n3])
        graph.add_edge(e := BasicEdge(n1, n2))
        with raises(TypeError):
            graph.substitute(n3, BasicEdge(n2, n1))
