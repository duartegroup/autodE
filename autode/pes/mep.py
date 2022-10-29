"""
Minimum energy path (MEP) finding
"""
import numpy as np
from typing import Tuple
from networkx.generators.lattice import grid_graph
from networkx.algorithms.shortest_paths import dijkstra_path


def peak_point(
    energies: np.ndarray,
    point1: Tuple[int, ...],
    point2: Tuple[int, ...],
) -> Tuple[int, ...]:
    """
    Find a point in an array of energies that connects two points via
    the minimum energy pathway. If there is no peak in the path then the
    highest energy point will be returned

    ---------------------------------------------------------------------------
    Arguments:
        energies: Tensor of energies

        point1: Indices of a point on the surface (len(point1) = energies.ndim)

        point2: Indices of another point on the surface

    Returns:
        (tuple(int, ...)):
    """

    def weight(u, v, d):
        """Weight between two nodes in the graph is taken as the abs diff"""
        return np.abs(energies[u] - energies[v])

    path = dijkstra_path(
        G=grid_graph(dim=energies.shape, periodic=False),
        source=point1,
        target=point2,
        weight=weight,
    )

    peak_idx = np.argmax(np.array([energies[p] for p in path]))

    return path[peak_idx]
