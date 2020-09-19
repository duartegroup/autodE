import networkx as nx
import numpy as np
from autode.log import logger
from autode.mol_graphs import is_isomorphic


def get_sum_energy_mep(saddle_point_r1r2, pes_2d):
    """
    Calculate the sum of the minimum energy path that traverses reactants (r)
    to products (p) via the saddle point (s)::

                /          p
               /     s
          r2  /
             /r
             ------------
                  r1

    Arguments:
        saddle_point_r1r2 (tuple(float)):

        pes_2d (autode.pes_2d.PES2d):

    Returns:
        (float): Path energy (Ha)
    """
    logger.info('Finding the total energy along the minimum energy pathway')

    reactant_point = (0, 0)
    product_point, product_energy = None, 9999

    # The saddle point indexes are those that are closest tp the saddle points
    # r1 and r2 distances
    saddle_point = (np.argmin(np.abs(pes_2d.r1s - saddle_point_r1r2[0])),
                    np.argmin(np.abs(pes_2d.r2s - saddle_point_r1r2[1])))

    # Generate a grid graph (i.e. all nodes are corrected
    energy_graph = nx.grid_2d_graph(pes_2d.n_points_r1, pes_2d.n_points_r2)

    min_energy = min([species.energy for species in pes_2d.species.flatten()])

    # For energy point on the 2D surface
    for i in range(pes_2d.n_points_r1):
        for j in range(pes_2d.n_points_r2):
            point_rel_energy = pes_2d.species[i, j].energy - min_energy

            # Populate the relative energy of each node in the graph
            energy_graph.nodes[i, j]['energy'] = point_rel_energy

            # Find the point where products are made
            if is_isomorphic(graph1=pes_2d.species[i, j].graph,
                             graph2=pes_2d.product_graph):

                # If products have not yet found, or they have and the energy
                # are lower but are still isomorphic
                if product_point is None or point_rel_energy < product_energy:
                    product_point = (i, j)
                    product_energy = point_rel_energy

    logger.info(f'Reactants at r1={pes_2d.r1s[0]:.4f} , '
                f'r2={pes_2d.r2s[0]:.4f} Å and '
                f'products r1={pes_2d.rs[product_point][0]:.4f}, '
                f'r2={pes_2d.rs[product_point][1]:.4f} Å')

    def energy_diff(curr_node, final_node, d):
        """Energy difference between the twp points on the graph. d is required
         to satisfy nx. Must only increase in energy to a saddle point so take
          the magnitude to prevent traversing s mistakenly"""
        return (np.abs(energy_graph.nodes[final_node]['energy']
                       - energy_graph.nodes[curr_node]['energy']))

    # Calculate the energy along the MEP up to the saddle point from reactants
    # and products
    path_energy = 0.0

    for point in (reactant_point, product_point):
        path_energy += nx.dijkstra_path_length(energy_graph,
                                               source=point,
                                               target=saddle_point,
                                               weight=energy_diff)

    logger.info(f'Path energy to {saddle_point} is {path_energy:.4f} Hd')
    return path_energy
