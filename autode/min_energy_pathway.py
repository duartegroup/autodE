import networkx as nx
import numpy as np
from autode.log import logger
from autode.mol_graphs import is_isomorphic
from autode.mol_graphs import make_graph


def get_sum_energy_mep(saddle_point_r1r2, pes_2d):
    """
    Calculate the sum of the minimum energy path that traverses reactants (r) to products (p) via the saddle point (s)

       |          p
       |     s
    r2 |
       | r
       ------------
            r1

    Arguments:
        saddle_point_r1r2 (tuple(float)):
        pes_2d (autode.pes_2d.PES2d):
    """
    logger.info('Finding the total energy along the minimum energy pathway')

    reactant_point = (0, 0)
    product_point, product_energy = None, 0

    # The saddle point indexes are those that are closest tp the saddle points r1 and r2 distances
    saddle_point = (np.argmin(np.abs(pes_2d.r1s - saddle_point_r1r2[0])),
                    np.argmin(np.abs(pes_2d.r2s - saddle_point_r1r2[1])))

    # Generate a grid graph (i.e. all nodes are corrected
    energy_graph = nx.grid_2d_graph(pes_2d.n_points_r1, pes_2d.n_points_r2)
    min_energy = min([species.energy for species in pes_2d.species.flatten()])

    # For energy point on the 2D surface
    for i in range(pes_2d.n_points_r1):
        for j in range(pes_2d.n_points_r2):

            # Populate the relative energy of each node in the graph
            energy_graph.nodes[i, j]['energy'] = pes_2d.species[i, j].energy - min_energy

            # Find the point where products are made
            make_graph(pes_2d.species[i, j])
            if is_isomorphic(graph1=pes_2d.species[i, j].graph, graph2=pes_2d.product_graph):

                # If products have not yet found, or they have and the energy are lower but are still isomorphic
                if product_point is None or pes_2d.species[i, j].energy < product_energy:
                    product_point = (i, j)

    def energy_diff(curr_node, final_node, d):
        return energy_graph.nodes[final_node]['energy'] - energy_graph.nodes[curr_node]['energy']

    # Calculate the energy along the MEP up to the saddle point from reactants and products
    rpath_energy = nx.dijkstra_path_length(energy_graph, source=reactant_point, target=saddle_point, weight=energy_diff)
    ppath_energy = nx.dijkstra_path_length(energy_graph, source=product_point, target=saddle_point, weight=energy_diff)

    logger.info(f'Path energy to {saddle_point} is {rpath_energy + ppath_energy:.4f} Hd')
    return rpath_energy + ppath_energy
