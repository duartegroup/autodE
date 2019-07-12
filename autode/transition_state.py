from .log import logger
from .config import Config
from .ORCAio import get_imag_frequencies_xyzs_energy_optts
from .single_point import get_single_point_energy
from .geom import calc_distance_matrix
from . import mol_graphs
from autode.templates import TStemplate


class TS(object):

    def make_graph(self):
        logger.info('Making TS graph with \'active\' edges')

        full_graph = mol_graphs.make_graph(self.xyzs, n_atoms=len(self.xyzs))
        distance_matrix = calc_distance_matrix(self.xyzs)

        for bond in self.active_bonds:
            atom_i, atom_j = bond
            full_graph.add_edge(atom_i, atom_j, active=True, weight=distance_matrix[atom_i, atom_j])

        nodes_to_keep = self.active_atoms.copy()
        for edge in full_graph.edges():
            node_i, node_j = edge
            if node_i in self.active_atoms:
                nodes_to_keep.append(node_j)
            if node_j in self.active_atoms:
                nodes_to_keep.append(node_i)

        self.graph = full_graph

        nodes_list = list(full_graph.nodes()).copy()
        truncated_graph = full_graph.copy()
        [truncated_graph.remove_node(node) for node in nodes_list if node not in nodes_to_keep]
        self.truncated_graph = truncated_graph

    def save_ts_template(self):
        logger.info('Saving TS template')
        try:
            ts_template = TStemplate(self.truncated_graph, reaction_class=self.reaction_class, solvent=self.solvent,
                                     charge=self.charge, mult=self.mult)
            ts_template.save_object()
            logger.info('Saved TS template')

        except ValueError or AttributeError:
            logger.error('Could not save TS template')

    def single_point(self):
        self.energy = get_single_point_energy(self, keywords=Config.sp_keywords, n_cores=Config.n_cores)

    def is_true_ts(self):
        if len(self.imag_freqs) == 1:
            return True
        else:
            return False

    def __init__(self, ts_guess, name='TS',converged=True):
        logger.info('Generating a TS object for {}'.format(name))

        self.name = name
        self.solvent = ts_guess.solvent
        self.charge = ts_guess.charge
        self.mult = ts_guess.mult
        self.converged = converged

        self.imag_freqs, self.xyzs, self.energy = get_imag_frequencies_xyzs_energy_optts(ts_guess.optts_out_lines)

        self.active_bonds = ts_guess.active_bonds
        self.active_atoms = list(set([atom_id for bond in self.active_bonds for atom_id in bond]))
        self.reaction_class = ts_guess.reaction_class

        self.graph = None
        self.truncated_graph = None
        self.make_graph()

        self.save_ts_template()
