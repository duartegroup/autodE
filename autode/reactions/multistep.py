from typing import Union
from autode.plotting import plot_reaction_profile
from autode.methods import get_hmethod
from autode.config import Config
from autode.reactions.reaction import Reaction
from autode.log import logger
from autode.utils import work_in


class MultiStepReaction:

    def __init__(self,
                 *args: 'autode.reactions.Reaction',
                 name:  str = 'reaction'
                 ):
        """
        Reaction with multiple steps.

        -----------------------------------------------------------------------
        Arguments:
            *args: Set of reactions to calculate the reaction profile for
        """
        self.name = str(name)
        self.reactions = self._checked_are_reactions(args)

    def calculate_reaction_profile(self,
                                   units: Union['autode.units.Unit', str] = 'kcal mol-1',
                                   ) -> None:
        """
        Calculate a multistep reaction profile using the products of step 1
        as the reactants of step 2 etc. Example

        .. code-block::

            >>> import autode as ade
            >>> ade.Config.n_cores = 16

            >>> ade.Config.ORCA.keywords.set_opt_basis_set('ma-def2-SVP')
            >>> ade.Config.ORCA.keywords.sp.basis_set = 'ma-def2-TZVP'

            >>> step1 = ade.Reaction('CC(F)=O.C[O-]>>CC([O-])(F)OC', solvent_name='water')
            >>> step2 = ade.Reaction('CC([O-])(F)OC>>CC(OC)=O.[F-]', solvent_name='water')

            >>> rxn = ade.MultiStepReaction(step1, step2)
            >>> rxn.calculate_reaction_profile()
        """
        logger.info(f'Calculating reaction profile for {self.n_steps} steps')

        @work_in(self.name)
        def calculate(rxn):

            rxn.find_lowest_energy_conformers()
            rxn.optimise_reacs_prods()
            rxn.locate_transition_state()
            rxn.find_lowest_energy_ts_conformer()
            rxn.calculate_single_points()

            return None

        # For all the reactions calculate the reactants, products and TS
        for i, r in enumerate(self.reactions):
            r.name = f'{self.name}_step{i}'
            prev_products = [] if i == 0 else self.reactions[i-1].prods

            calculate(rxn=r, _prev_products=prev_products)

        self._balance()
        plot_reaction_profile(self.reactions, units=units, name=self.name)
        return None

    @property
    def n_steps(self) -> int:
        """Number of steps in the multistep reaction"""
        return len(self.reactions)

    def _balance(self) -> None:
        """
        Balance reactants and products in each reaction such that the total
        number of atoms is conserved throughout the reaction. Allows for
        e.g. catalytic cycles to be evaluated and plotted
        """

        for i in range(self.n_steps - 1):
            j = i + 1   # Index of the next step in the sequence of reactions

            if self.reactions[j].has_identical_composition_as(self.reactions[i]):
                continue

            if self._adds_molecule(i, j):
                self._add_mol_to_reacs_prods(mol=self._added_molecule(i, j),
                                             to_idx=j)

            if self._adds_molecule(j, i):
                self._add_mol_to_reacs_prods(mol=self._added_molecule(j, i),
                                             from_idx=j)
        return None

    def _adds_molecule(self,
                       step_idx:      int,
                       next_step_idx: int
                       ) -> bool:
        """Does reaction 2 have more molecules(atoms) than reaction 1"""

        def total_n_atoms(rxn):
            return sum(m.n_atoms for m in rxn.reacs)

        return (total_n_atoms(self.reactions[step_idx])
                < total_n_atoms(self.reactions[next_step_idx]))

    def _add_mol_to_reacs_prods(self, mol, from_idx=0, to_idx=None):
        """Add a molecule to both reactants and products"""
        if to_idx is None:
            to_idx = self.n_steps

        for _i, _reaction in enumerate(self.reactions[from_idx:to_idx]):
            _reaction.reacs.append(mol.to_reactant())
            _reaction.prods.append(mol.to_product())

        return None

    def _added_molecule(self,
                        step_idx:      int,
                        next_step_idx: int
                        ) -> 'autode.species.molecule.Molecule':
        """
        Extract the added molecule going from a step to the next. For example,

        A + B -> C
        C + D -> E

        the added molecule would be C.
        -----------------------------------------------------------------------
        Arguments:
            step_idx:
            next_step_idx:

        Returns:
            (autode.species.molecule.Molecule):

        Raises:
            (RuntimeError):
        """
        prods = self.reactions[step_idx].prods

        for mol in self.reactions[next_step_idx].reacs:
            if any(mol.atomic_symbols == p.atomic_symbols for p in prods):
                continue

            return mol

        raise RuntimeError(f'Failed to find the added molecule')

    @staticmethod
    def _checked_are_reactions(args):
        """Check that all objects in a list are autodE Reaction instances"""

        if not all(isinstance(item, Reaction) for item in args):
            raise ValueError('Cannot initialise a multistep reaction from: '
                             f'{args}. Must all be autode Reaction instances')

        return args
