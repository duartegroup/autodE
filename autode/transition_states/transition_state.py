from autode.log import logger
from autode.transition_states.base import get_displaced_atoms_along_mode
from autode.config import Config
from autode.transition_states.templates import TStemplate
from autode.utils import requires_atoms
from autode.exceptions import AtomsNotFound, NoNormalModesFound
from autode.calculation import Calculation
from autode.methods import get_hmethod
from autode.conformers.conformers import Conformer
from autode.conformers.conf_gen import get_simanl_atoms
from autode.transition_states.base import TSbase


class TransitionState(TSbase):

    def _run_opt_ts_calc(self, method):

        self.optts_calc = Calculation(name=f'{self.name}_optts', molecule=self, method=method, n_cores=Config.n_cores,
                                      keywords_list=method.keywords.opt_ts,
                                      bond_ids_to_add=self.bond_rearrangement.all,
                                      other_input_block=method.keywords.optts_block)
        self.optts_calc.run()

        # Attempt to set the frequencies, atoms and energy from the calculation
        try:
            self.imaginary_frequencies = self.optts_calc.get_imag_freqs()
            self.set_atoms(atoms=self.optts_calc.get_final_atoms())
            self.energy = self.optts_calc.get_energy()

        except (AtomsNotFound, NoNormalModesFound):
            logger.error('Transition state optimisation calculation failed')

        return

    @requires_atoms()
    def opt_ts(self):
        """Optimise this TS to a true TS """

        self._run_opt_ts_calc(method=get_hmethod())

        # A transition state is a first order saddle point i.e. has a single imaginary frequency
        if len(self.imaginary_frequencies) == 1:
            logger.info('Found a TS with a single imaginary frequency')
            return None

        if len(self.imaginary_frequencies) == 0:
            logger.error('Transition state optimisation did not return any imaginary frequencies')
            return None

        # There is more than one imaginary frequency. Will assume that the most negative is the correct mode..

        for disp_magnitude in [1, -1]:
            self.atoms = get_displaced_atoms_along_mode(self.optts_calc, mode_number=7, disp_magnitude=disp_magnitude)
            self._run_opt_ts_calc(method=get_hmethod())

            if len(self.imaginary_frequencies) == 1:
                logger.info('Displacement along second imaginary mode successful. Now have 1 imaginary mode')
                break

        return None

    def is_true_ts(self):
        """Is this TS a 'true' TS i.e. has at least on imaginary mode in the hessian"""
        return True if len(self.imaginary_frequencies) > 0 else False

    def save_ts_template(self, folder_path=None):
        """Save a transition state template containing the active bond lengths, solvent_name and charge in folder_path

        Keyword Arguments:
            folder_path (str): folder to save the TS template to (default: {None})
        """
        logger.info('Saving TS template')

        # TODO fix this

        try:
            ts_template = TStemplate(self.truncated_graph, solvent=self.solvent,  charge=self.charge, mult=self.mult)
            ts_template.save_object(folder_path=folder_path)
            logger.info('Saved TS template')

        except (ValueError, AttributeError):
            logger.error('Could not save TS template')

    def find_lowest_energy_conformer(self):
        """For a transition state object find the lowest energy conformer"""

        # TODO check isomorphism

        # TODO check that energy doesn't rise

        # TODO set the final optts calc

        return None

    def __init__(self, ts_guess, bond_rearrangement):
        """
        Transition State

        Arguments:
            ts_guess (autode.transition_states.ts_guess.TSguess)
            bond_rearrangement (autode.bond_rearrangement.BondRearrangement):
            name (str):
        """
        super().__init__(atoms=ts_guess.atoms, reactant=ts_guess.reactant,
                         product=ts_guess.product, name=f'TS_{ts_guess.name}')

        self.bond_rearrangement = bond_rearrangement
        self.conformers = None

        self.optts_calc = None
        self.imaginary_frequencies = []


class SolvatedTransitionState(TransitionState):

    def __init__(self, ts_guess, bond_rearrangement, name='TS'):

        super().__init__(ts_guess=ts_guess, bond_rearrangement=bond_rearrangement, name=name)
        self.qm_solvent_xyzs = ts_guess.qm_solvent_xyzs
        self.mm_solvent_xyzs = ts_guess.mm_solvent_xyzs

        # self.charges = ts_guess.get_charges()[:self.n_atoms]
