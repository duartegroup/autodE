from autode.log import logger
from autode.constants import Constants
from autode.config import Config
from autode.mol_graphs import is_isomorphic
from autode.transition_states.templates import TStemplate
from autode.calculation import Calculation
from autode.species import Species
from autode.transition_states.optts import get_ts
from autode.conformers.conformers import Conformer
from autode.conformers.conf_gen import get_simanl_atoms
import numpy as np
from autode.transition_states.base import TSbase
from autode.solvent.explicit_solvent import do_explicit_solvent_qmmm


class TransitionState(TSbase):

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

    def check_optts_convergence(self):

        if not self.optts_calc.optimisation_converged():
            if self.optts_calc.optimisation_nearly_converged():
                logger.info('OptTS nearly did converge. Will try more steps')
                self.optts_nearly_converged = True
                all_xyzs = self.optts_calc.get_final_atoms()
                self.xyzs = all_xyzs[:self.n_atoms]
                if self.qm_solvent_xyzs is not None:
                    self.qm_solvent_xyzs = all_xyzs[self.n_atoms:]
                self.name += '_reopt'
                self.run_optts()
                return

            logger.warning('OptTS calculation was no where near converging')

        else:
            self.optts_converged = True

        return

    def save_ts_template(self, folder_path=None):
        """Save a transition state template containing the active bond lengths, solvent_name and charge in folder_path

        Keyword Arguments:
            folder_path (str): folder to save the TS template to (default: {None})
        """
        logger.info('Saving TS template')
        try:
            ts_template = TStemplate(self.truncated_graph, reaction_type=self.reaction_class, solvent=self.solvent,
                                     charge=self.charge, mult=self.mult)
            ts_template.save_object(folder_path=folder_path)
            logger.info('Saved TS template')

        except (ValueError, AttributeError):
            logger.error('Could not save TS template')

    def do_displacements(self, size=0.75):
        """Attempts to remove second imaginary mode by displacing either way along it

        Keyword Arguments:
            size (float): magnitude to displace along (default: {0.75})
        """
        mode_lost = False
        imag_freqs = []
        orig_optts_calc = deepcopy(self.optts_calc)
        orig_name = copy(self.name)
        self.xyzs = get_displaced_xyzs_along_imaginary_mode(self.optts_calc, self.n_atoms, displacement_magnitude=size)
        self.name += '_dis'
        self.run_optts()
        if self.calc_failed:
            logger.error('Displacement lost correct imaginary mode, trying backwards displacement')
            mode_lost = True
            self.calc_failed = False

        if not mode_lost:
            self.check_optts_convergence()
            if not self.calc_failed:
                imag_freqs, _, _ = self.get_imag_frequencies_xyzs_energy()
                if len(imag_freqs) > 1:
                    logger.warning(f'OptTS calculation returned {len(imag_freqs)} imaginary frequencies, '
                                   f'trying displacement backwards')
                if len(imag_freqs) == 1:
                    logger.info('Displacement fixed multiple imaginary modes')
                    return
            else:
                logger.error('Displacement lost correct imaginary mode, trying backwards displacement')
                mode_lost = True

        if len(imag_freqs) > 1 or mode_lost:
            self.optts_calc = orig_optts_calc
            self.name = orig_name
            self.xyzs = get_displaced_xyzs_along_imaginary_mode(self.optts_calc, self.n_atoms,
                                                                displacement_magnitude=-size)
            self.name += '_dis2'
            self.run_optts()
            if self.calc_failed:
                logger.error('Displacement lost correct imaginary mode')
                self.calc_failed = True
                return

            imag_freqs, _, _ = self.get_imag_frequencies_xyzs_energy()

            if len(imag_freqs) > 1:
                logger.error('Couldn\'t remove optts_block imaginary frequencies by displacement')

        return

    def run_optts(self, imag_freq_threshold=-50):
        """Runs the optts calc. Calculates a hessian first to the ts guess has the right imaginary mode

        Args:
            imag_freq_threshold (int, optional): If the imaginary mode has a higher (less negative) frequency than this it will not count as the right mode. Defaults to -50.
        """
        logger.info('Getting orca out lines from OptTS calculation')

        if self.qm_solvent_xyzs is not None:
            solvent_atoms = [i for i in range(self.n_atoms, self.n_atoms + len(self.qm_solvent_xyzs))]
        else:
            solvent_atoms = None

        self.hess_calc = Calculation(name=self.name + '_hess', molecule=self, method=self.method,
                                     keywords_list=self.method.keywords.hess, n_cores=Config.n_cores,
                                     partial_hessian=solvent_atoms)

        self.hess_calc.run()

        imag_freqs = self.hess_calc.get_imag_freqs()
        if len(imag_freqs) == 0:
            logger.info('Hessian showed no imaginary modes')
            self.calc_failed = True
            return
        if len(imag_freqs) > 1:
            logger.warning(f'Hessian had {len(imag_freqs)} imaginary modes')
        if imag_freqs[0] > imag_freq_threshold:
            logger.info('Imaginary modes were too small to be significant')
            self.calc_failed = True
            return

        if not ts_has_correct_imaginary_vector(self.hess_calc, n_atoms=self.n_atoms, active_bonds=self.active_bonds,
                                               threshold_contribution=0.1):
            self.calc_failed = True
            return

        self.optts_calc = Calculation(name=self.name + '_optts', molecule=self, method=self.method,
                                      keywords_list=self.method.keywords.opt_ts, n_cores=Config.n_cores,
                                      bond_ids_to_add=self.active_bonds,
                                      partial_hessian=solvent_atoms, other_input_block=self.method.keywords.optts_block,
                                      cartesian_constraints=solvent_atoms)

        self.optts_calc.run()
        all_xyzs = self.optts_calc.get_final_atoms()
        self.xyzs = all_xyzs[:self.n_atoms]
        if self.qm_solvent_xyzs is not None:
            self.qm_solvent_xyzs = all_xyzs[self.n_atoms:]
        return

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
        if method is None:
            method = self.method

        if solvent_mol:
            point_charges = []
            for i, xyz in enumerate(self.mm_solvent_xyzs):
                point_charges.append(xyz + [solvent_mol.charges[i % solvent_mol.n_atoms]])
        else:
            point_charges = None

        sp = Calculation(name=self.name + '_sp', molecule=self, method=method, keywords_list=self.method.sp_keywords,
                         n_cores=Config.n_cores, max_core_mb=Config.max_core, charges=self.point_charges)
        sp.run()
        self.energy = sp.get_energy()

    def generate_conformers(self):

        self.conformers = []

        bond_list = list(self.graph.edges)
        conf_xyzs = get_simanl_atoms(name=self.name, init_xyzs=self.xyzs, bond_list=bond_list, stereocentres=self.stereocentres, dist_consts=self.dist_consts.copy())

        for i in range(len(conf_xyzs)):

            self.conformers.append(Conformer(name=self.name + f'_conf{i}', xyzs=conf_xyzs[i], solvent_name=self.solvent,
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

    def opt_ts(self, solvent_mol):
        """Run the optts calculation

        Returns:
            ts object: the optimised transition state conformer
        """
        name = self.name

        ts_conf_get_ts_output = get_ts(self, solvent_mol)
        if ts_conf_get_ts_output is None:
            return None

        self.converged = ts_conf_get_ts_output[1]
        self.energy = self.optts_calc.get_energy()

        self.name = name

        return self

    def find_lowest_energy_conformer(self, solvent_mol):
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

        if solvent_mol is not None:
            _, qmmm_xyzs, n_qm_atoms = do_explicit_solvent_qmmm(self, self.solvent, self.method)

            self.xyzs = qmmm_xyzs[:self.n_atoms]
            self.qm_solvent_xyzs = qmmm_xyzs[self.n_atoms: n_qm_atoms]
            self.mm_solvent_xyzs = qmmm_xyzs[n_qm_atoms:]

        return self.opt_ts(solvent_mol)

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
        self.stereocentres = ts_guess.stereocentres
        self.n_atoms = ts_guess.n_atoms
        self.reactant = ts_guess.reactant
        self.product = ts_guess.product
        self.xyzs = ts_guess.xyzs
        self.qm_solvent_xyzs = ts_guess.qm_solvent_xyzs
        self.mm_solvent_xyzs = ts_guess.mm_solvent_xyzs

        self.imag_freqs, _, self.energy = ts_guess.get_imag_frequencies_xyzs_energy()
        self.charges = ts_guess.get_charges()[:self.n_atoms]

        self.active_bonds = ts_guess.active_bonds
        self.active_atoms = list(set([atom_id for bond in self.active_bonds for atom_id in bond]))
        self.reaction_class = ts_guess.reaction_class

        self.graph = None
        self.truncated_graph = None
        self.graph_from_xyzs = None
        self.make_graph()

        self.dist_consts = None
        self.get_dist_consts()

        self.conformers = None
        self.n_conformers = None

        if Config.make_ts_template:
            self.save_ts_template()

        self.optts_converged = False
        self.optts_nearly_converged = False
        self.optts_calc = None
        self.hess_calc = None

        self.calc_failed = False

        self.point_charges = None
