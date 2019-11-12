from autode.log import logger
from autode.config import Config
from autode.geom import calc_distance_matrix
from autode.geom import xyz2coord
from autode import mol_graphs
from autode.transition_states.templates import TStemplate
from autode.calculation import Calculation
from autode.transition_states.ts_conformers import rot_bond
from autode.transition_states.rot_fragments import RotFragment
import numpy as np


class TS:

    def make_graph(self):
        logger.info('Making TS graph with \'active\' edges')

        full_graph = mol_graphs.make_graph(self.xyzs, n_atoms=len(self.xyzs))
        distance_matrix = calc_distance_matrix(self.xyzs)

        for bond in self.active_bonds:
            atom_i, atom_j = bond
            full_graph.add_edge(atom_i, atom_j, active=True,
                                weight=distance_matrix[atom_i, atom_j])

        nodes_to_keep = self.active_atoms.copy()
        for edge in full_graph.edges():
            node_i, node_j = edge
            if node_i in self.active_atoms:
                nodes_to_keep.append(node_j)
            if node_j in self.active_atoms:
                nodes_to_keep.append(node_i)

        self.graph = full_graph

        graph_with_fbonds = self.graph.copy()
        for bond in self.active_bonds:
            graph_with_fbonds.add_edge(*bond)

        self.graph_with_fbonds = graph_with_fbonds

        nodes_list = list(full_graph.nodes()).copy()
        truncated_graph = full_graph.copy()
        [truncated_graph.remove_node(
            node) for node in nodes_list if node not in nodes_to_keep]
        self.truncated_graph = truncated_graph

    def save_ts_template(self, folder_path=None):
        """
        Save a transition state template containing the active bond lengths, solvent and charge in folder_path

        :param folder_path: (str) folder to save the TS template to
        :return:
        """
        logger.info('Saving TS template')
        try:
            ts_template = TStemplate(self.truncated_graph, reaction_class=self.reaction_class, solvent=self.solvent,
                                     charge=self.charge, mult=self.mult)
            ts_template.save_object(folder_path=folder_path)
            logger.info('Saved TS template')

        except (ValueError, AttributeError):
            logger.error('Could not save TS template')

    def get_coords(self):
        return xyz2coord(self.xyzs)

    def get_atom_label(self, atom_i):
        return self.xyzs[atom_i][0]

    def get_bonded_atoms_to_i(self, atom_i):
        bonded_atoms = []
        for edge in self.graph.edges():
            if edge[0] == atom_i:
                bonded_atoms.append(edge[1])
            if edge[1] == atom_i:
                bonded_atoms.append(edge[0])
        return bonded_atoms

    def get_dist_consts(self):
        dist_const_dict = {}
        for bond in self.active_bonds:
            atom1, atom2 = bond
            coords = self.get_coords()
            bond_length = np.linalg.norm(coords[atom1] - coords[atom2])
            dist_const_dict[bond] = bond_length
        self.dist_consts = dist_const_dict

    def is_true_ts(self):
        if len(self.imag_freqs) == 1:
            return True
        else:
            return False

    def single_point(self, method=None):
        logger.info(f'Running single point energy evaluation of {self.name}')

        sp = Calculation(name=self.name + '_sp', molecule=self, method=self.method if method is None else method,
                         keywords=self.method.sp_keywords, n_cores=Config.n_cores, max_core_mb=Config.max_core)
        sp.run()
        self.energy = sp.get_energy()

    def get_rotatable_bonds(self):
        """Looks to see if each bond is worth rotating, i.e not methyl, trifluoromethyl or linear

        Returns:
            list -- list of bond tuples 
        """
        logger.info('Finding the bonds worth rotating')
        bonds = list(self.graph_with_fbonds.edges)
        bonds_worth_rotating = []

        for potential_bond in bonds:
            if self.pi_bonds is not None:
                if potential_bond in self.pi_bonds:
                    continue
            suitable_bond = True
            for index, atom in enumerate(potential_bond):
                bonded_atoms = self.get_bonded_atoms_to_i(atom)
                bonded_atoms.remove(potential_bond[1-index])
                # don't rotate terminal atoms
                if len(bonded_atoms) == 0:
                    suitable_bond = False
                # don't rotate nitrile, methyl or trifluoromethyl
                if len(bonded_atoms) == 1 and self.get_atom_label(atom) == 'C':
                    if self.get_atom_label(bonded_atoms[0]) == 'N':
                        suitable_bond = False
                if all([self.get_atom_label(i) == 'H' for i in bonded_atoms]):
                    suitable_bond = False
                if all([self.get_atom_label(i) == 'F' for i in bonded_atoms]):
                    suitable_bond = False
            if suitable_bond:
                # don't rotate rings
                if not mol_graphs.bond_in_cycle(self.graph_with_fbonds, potential_bond):
                    bonds_worth_rotating.append(potential_bond)
        logger.info(f'Found {len(bonds_worth_rotating)} bonds worth rotating')
        self.rotatable_bonds = bonds_worth_rotating

    def get_central_bond(self):
        logger.info('Getting central bond')
        split_graphs_indices = [mol_graphs.split_mol_across_bond(
            self.graph_with_fbonds, [bond]) for bond in self.rotatable_bonds]
        smallest_difference = len(self.xyzs)
        for index, split in enumerate(split_graphs_indices):
            difference = abs(len(split[0]) - len(split[1]))
            if difference < smallest_difference:
                smallest_difference = difference
                central_bond = self.rotatable_bonds[index]
        logger.info(f'Found central bond {central_bond}')
        self.central_bond = central_bond

    def decompose(self):
        logger.info('Decomposing molecule into rotatable fragments')

        rot_bonds = self.rotatable_bonds.copy()
        rot_bonds.remove(self.central_bond)

        base_atom = self.central_bond[0]

        frag_level = 0
        rot_frags = []
        parent_graph = self.graph.copy()
        atoms_in_frags = []
        while rot_bonds:

            split_graphs_indices = [mol_graphs.split_mol_across_bond(
                parent_graph, [bond]) for bond in rot_bonds]
            largest_difference = None
            edge_bonds = []
            all_fragment_atoms = []
            for index, split in enumerate(split_graphs_indices):
                if base_atom in split[0]:
                    difference = len(split[0]) - len(split[1])
                    fragment_atoms = split[1]
                else:
                    difference = len(split[1]) - len(split[0])
                    fragment_atoms = split[0]
                if largest_difference is None:
                    largest_difference = difference
                    edge_bonds = [rot_bonds[index]]
                    all_fragment_atoms = [fragment_atoms]
                elif difference > largest_difference:
                    largest_difference = difference
                    edge_bonds = [rot_bonds[index]]
                    all_fragment_atoms = [fragment_atoms]
                elif difference == largest_difference:
                    edge_bonds.append(rot_bonds[index])
                    all_fragment_atoms.append(fragment_atoms)
            for edge_bond in edge_bonds:
                rot_bonds.remove(edge_bond)
            for edge_bond, frag_atoms in zip(edge_bonds, all_fragment_atoms):
                rot_frags.append(RotFragment(ts=self, rot_bond=edge_bond, atoms=frag_atoms,
                                             base_atom=base_atom, parent_graph=parent_graph, level=frag_level, all_rot_bonds=rot_bonds))
                atoms_in_frags += frag_atoms
            all_bonds_split_graphs = mol_graphs.split_mol_across_bond(
                parent_graph, edge_bonds, return_graphs=True)
            for graph in all_bonds_split_graphs:
                if base_atom in list(graph.nodes):
                    parent_graph = graph
                    break
            frag_level += 1
        rot_frags.append(RotFragment(ts=self, rot_bond=self.central_bond, level=frag_level, atoms=[
                         atom for atom in range(len(self.xyzs)) if atom not in atoms_in_frags]))
        frag_level += 1

        logger.info(f'Made {len(rot_frags)} rotatable fragments')

        # order frags by frag_level
        ordered_frags = []
        for i in range(frag_level):
            for frag in rot_frags:
                if frag.level == i:
                    ordered_frags.append(frag)

        self.rot_frags = ordered_frags

    def rotate(self):
        # go from the outside in
        logger.info('Rotating fragments')
        for frag in self.rot_frags:
            frag.rotate()

    def opt_ts(self):
        opt = Calculation(name=self.name + '_opt', molecule=self, method=self.method, keywords=self.method.opt_keywords,
                          n_cores=Config.n_cores, max_core_mb=Config.max_core, distance_constraints=self.dist_consts)
        opt.run()
        self.energy = opt.get_energy()
        self.xyzs = opt.get_final_xyzs()

        self.hess_calc = Calculation(name=self.name + '_hess', molecule=self, method=self.method,
                                     keywords=self.method.hess_keywords, n_cores=Config.n_cores, max_core_mb=Config.max_core)

        self.hess_calc.run()

        self.optts_calc = Calculation(name=self.name + '_optts', molecule=self, method=self.method,
                                      keywords=self.method.opt_ts_keywords, n_cores=Config.n_cores,
                                      max_core_mb=Config.max_core, bond_ids_to_add=self.active_bonds,
                                      optts_block=self.method.opt_ts_block)

        self.optts_calc.run()
        self.xyzs = self.optts_calc.get_final_xyzs()

    def do_conformers(self):
        self.get_rotatable_bonds()
        if len(self.rotatable_bonds) == 0:
            logger.info('No bonds to rotate')
            return
        self.get_central_bond()
        self.decompose()
        self.rotate()
        self.opt_ts()

    def __init__(self, ts_guess, name='TS', converged=True):
        logger.info(f'Generating a TS object for {name}')

        self.name = name
        self.solvent = ts_guess.solvent
        self.charge = ts_guess.charge
        self.mult = ts_guess.mult
        self.converged = converged
        self.method = ts_guess.method
        self.pi_bonds = ts_guess.pi_bonds

        self.imag_freqs, self.xyzs, self.energy = ts_guess.get_imag_frequencies_xyzs_energy()

        self.active_bonds = ts_guess.active_bonds
        self.active_atoms = list(
            set([atom_id for bond in self.active_bonds for atom_id in bond]))
        self.reaction_class = ts_guess.reaction_class

        self.graph = None
        self.truncated_graph = None
        self.graph_with_fbonds = None
        self.make_graph()

        self.rotatable_bonds = []
        self.central_bond = None
        self.rot_fragments = []

        self.dist_consts = None
        self.get_dist_consts()

        self.conformers = None
        self.n_conformers = None

        if Config.make_ts_template:
            self.save_ts_template()
