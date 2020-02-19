from autode.log import logger
from autode.constants import Constants
from autode.config import Config
from autode.geom import calc_distance_matrix
from autode.mol_graphs import make_graph
from autode.mol_graphs import is_isomorphic
from autode.transition_states.templates import TStemplate
from autode.calculation import Calculation
from autode.transition_states.ts_guess import TSguess
from autode.transition_states.optts import get_ts
from autode.conformers.conformers import Conformer
from autode.conformers.conf_gen import gen_simanl_conf_xyzs
import numpy as np

from autode.solvent.explicit_solvent import do_explicit_solvent_qmmm


class TS(TSguess):

    def make_graph(self):
        logger.info('Making TS graph with \'active\' edges')

        full_graph = make_graph(self.xyzs, n_atoms=len(self.xyzs))
        distance_matrix = calc_distance_matrix(self.xyzs)

        self.graph_from_xyzs = full_graph.copy()

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

        nodes_list = list(full_graph.nodes()).copy()
        truncated_graph = full_graph.copy()
        [truncated_graph.remove_node(node) for node in nodes_list if node not in nodes_to_keep]
        self.truncated_graph = truncated_graph

    def save_ts_template(self, folder_path=None):
        """Save a transition state template containing the active bond lengths, solvent and charge in folder_path

        Keyword Arguments:
            folder_path (str): folder to save the TS template to (default: {None})
        """
        logger.info('Saving TS template')
        try:
            ts_template = TStemplate(self.truncated_graph, reaction_class=self.reaction_class, solvent=self.solvent,
                                     charge=self.charge, mult=self.mult)
            ts_template.save_object(folder_path=folder_path)
            logger.info('Saved TS template')

        except (ValueError, AttributeError):
            logger.error('Could not save TS template')

    def get_atom_label(self, atom_i):
        return self.xyzs[atom_i][0]

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

    def single_point(self, solvent_mol, method=None):
        logger.info(f'Running single point energy evaluation of {self.name}')

        if solvent_mol:
            self.energy, _, _ = do_explicit_solvent_qmmm(self, solvent_mol, method, hlevel=True, fix_qm=True)
        else:
            sp = Calculation(name=self.name + '_sp', molecule=self, method=self.method if method is None else method,
                             keywords=self.method.sp_keywords, n_cores=Config.n_cores, max_core_mb=Config.max_core)
            sp.run()
            self.energy = sp.get_energy()

    def generate_conformers(self):

        self.conformers = []

        bond_list = list(self.graph.edges)
        conf_xyzs = gen_simanl_conf_xyzs(name=self.name, init_xyzs=self.xyzs, bond_list=bond_list, stereocentres=self.stereocentres, dist_consts=self.dist_consts.copy())

        for i in range(len(conf_xyzs)):

            self.conformers.append(Conformer(name=self.name + f'_conf{i}', xyzs=conf_xyzs[i], solvent=self.solvent,
                                             charge=self.charge, mult=self.mult, dist_consts=self.dist_consts))

        self.n_conformers = len(self.conformers)

    def strip_non_unique_confs(self, energy_threshold_kj=1):
        logger.info('Stripping conformers with energy ∆E < 1 kJ mol-1 to others')
        # conformer.energy is in Hartrees
        d_e = energy_threshold_kj / Constants.ha2kJmol

        # The first conformer must be unique
        unique_conformers = [self.conformers[0]]

        for i in range(1, self.n_conformers):
            unique = True
            for j in range(len(unique_conformers)):
                if self.conformers[i].energy - d_e < self.conformers[j].energy < self.conformers[i].energy + d_e:
                    unique = False
                    break
            if unique:
                unique_conformers.append(self.conformers[i])

        logger.info(f'Stripped {self.n_conformers - len(unique_conformers)} conformers from a total of {self.n_conformers}')
        self.conformers = unique_conformers
        self.n_conformers = len(self.conformers)

    def strip_confs_failed(self):
        self.conformers = [conf for conf in self.conformers if conf.xyzs is not None and conf.energy is not None]
        self.n_conformers = len(self.conformers)

    def opt_ts(self):
        """Run the optts calculation

        Returns:
            ts object: the optimised transition state conformer
        """
        name = self.name

        ts_conf_get_ts_output = get_ts(self)
        if ts_conf_get_ts_output is None:
            return None

        self.converged = ts_conf_get_ts_output[1]
        self.energy = self.optts_calc.get_energy()

        self.name = name

        return self

    def find_lowest_energy_conformer(self):
        """For a transition state object find the lowest conformer in energy and set it as the mol.xyzs and mol.energy

        Returns:
            ts object: optimised ts object
        """
        self.generate_conformers()
        [self.conformers[i].optimise() for i in range(len(self.conformers))]
        self.strip_confs_failed()
        self.strip_non_unique_confs()
        [self.conformers[i].optimise(method=self.method)
         for i in range(len(self.conformers))]

        lowest_energy = None
        for conformer in self.conformers:
            if conformer.energy is None:
                continue

            conformer_graph = make_graph(conformer.xyzs, self.n_atoms)

            if is_isomorphic(self.graph_from_xyzs, conformer_graph):
                # If the conformer retains the same connectivity
                if lowest_energy is None:
                    lowest_energy = conformer.energy

                elif conformer.energy <= lowest_energy:
                    self.energy = conformer.energy
                    self.xyzs = conformer.xyzs
                    lowest_energy = conformer.energy

                else:
                    pass
            else:
                logger.warning('Conformer had a different molecular graph. Ignoring')

        logger.info('Set lowest energy conformer energy & geometry as mol.energy & mol.xyzs')

        return self.opt_ts()

    def __init__(self, ts_guess=None, name='TS', converged=True):
        logger.info(f'Generating a TS object for {name}')

        self.name = name

        if ts_guess is None:
            logger.error('A TS needs to be initialised from a TSguess object')
            return

        self.solvent = ts_guess.solvent
        self.charge = ts_guess.charge
        self.mult = ts_guess.mult
        self.converged = converged
        self.method = ts_guess.method

        self.reactant = ts_guess.reactant
        self.product = ts_guess.product

        self.imag_freqs, self.xyzs, self.energy = ts_guess.get_imag_frequencies_xyzs_energy()
        self.n_atoms = len(self.xyzs)

        self.active_bonds = ts_guess.active_bonds
        self.active_atoms = list(set([atom_id for bond in self.active_bonds for atom_id in bond]))
        self.reaction_class = ts_guess.reaction_class

        self.graph = None
        self.truncated_graph = None
        self.graph_from_xyzs = None
        self.make_graph()

        self.dist_consts = None
        self.get_dist_consts()

        self.stereocentres = None
        self.conformers = None
        self.n_conformers = None

        if Config.make_ts_template:
            self.save_ts_template()

        self.optts_converged = False
        self.optts_nearly_converged = False
        self.optts_calc = None
        self.hess_calc = None

        self.calc_failed = False
