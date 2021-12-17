"""Module dedicated to print world objects."""
from subprocess import CompletedProcess, run
from tempfile import NamedTemporaryFile
from typing import IO, Dict, Type

from simplifier.world.edges import DefinitionEdge, OperandEdge, WorldRelation
from simplifier.world.nodes import Operation, WorldObject
from simplifier.world.world import World


class WorldPlotter:
    """Class in charge of generating dotviz graphs from World objects."""

    EDGE_DECORATION: Dict[Type[WorldRelation], Dict[str, str]] = {OperandEdge: {"color": "blue"}, DefinitionEdge: {"color": "red"}}

    def __init__(self, buffer: IO):
        """Initialize the plotter with the given buffer."""
        self._buffer = buffer

    @property
    def buffer(self) -> IO:
        """Return the buffered stream."""
        self._buffer.flush()
        self._buffer.seek(0)
        return self._buffer

    def plot(self, world: World):
        """Plot the given World to the current buffer."""
        self._buffer.write("digraph {\n")
        for node in world._graph.nodes:
            self._buffer.write(self._declare_node(node))
        for edge in world._graph.edges:
            self._buffer.write(self._declare_edge(edge))
        self._buffer.write("}")

    def _declare_edge(self, edge: WorldRelation) -> str:
        """Declare the given edge in dotviz format."""
        return f"""{hash(edge.source)} -> {hash(edge.sink)} [{' '.join([f'{key}="{value}";' for key, value in self.EDGE_DECORATION.get(edge.__class__, {}).items()])}];"""

    def _declare_node(self, node: WorldObject) -> str:
        """Declare the given node in dotviz format."""
        decoration = {}
        if isinstance(node, Operation):
            decoration["label"] = node.SYMBOL
        else:
            decoration["label"] = str(node)
        return f"""{hash(node)} [{' '.join([f'{key}="{value}";' for key, value in decoration.items()])}];"""


# Helper methods for convenient usage.


def world_to_graphviz(world: World, buffer: IO) -> IO:
    """Generate a dotviz string representation of the given world object."""
    plotter = WorldPlotter(buffer)
    plotter.plot(world)
    return plotter.buffer


def world_to_dotfile(world: World, path: str):
    """Generate a dotviz file at the given location from the given world object."""
    with open(path, "w+") as buffer:
        world_to_graphviz(world, buffer)


def world_to_string(world: World) -> str:
    """Generate an ascii representation from the given world object utilizing graph-easy."""
    with NamedTemporaryFile(mode="w+") as buffer:
        world_to_graphviz(world, buffer)
        result: CompletedProcess = run(["graph-easy", "--as=ascii", buffer.name], capture_output=True)
    if result.stderr:
        return result.stderr.decode("utf-8")
    return result.stdout.decode("utf-8")


def print_world(world):
    """Print an ascii representation of the given world."""
    print(world_to_string(world))
