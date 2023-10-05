from typing import Union, TYPE_CHECKING

from autode.plotting import plot_reaction_profile
from autode.methods import get_hmethod
from autode.config import Config
from autode.reactions.reaction import Reaction
from autode.log import logger
from autode.utils import work_in

if TYPE_CHECKING:
    from autode.units import Unit
    from autode.species.molecule import Molecule


class MultiStepReaction:
    def __init__(self, *args: "Reaction", name: str = "reaction"):
        """
        Reaction with multiple steps.

        -----------------------------------------------------------------------
        Arguments:
            *args: Set of reactions to calculate the reaction profile for
        """
        self.name = str(name)
        self.reactions = self._checked_are_reactions(args)

    def calculate_reaction_profile(
        self,
        units: Union["Unit", str] = "kcal mol-1",
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
        logger.info(f"Calculating an {self.n_steps} step reaction profile")
        hmethod = get_hmethod() if Config.hmethod_conformers else None

        @work_in(self.name)
        def calculate(idx):
            rxn = self.reactions[idx]

            if idx == 0:  # First reaction
                rxn.find_lowest_energy_conformers()
                rxn.optimise_reacs_prods()

            else:  # Subsequent reactions have already calculated components
                self._set_reactants_from_previous_products(idx)

                for prod in rxn.prods:
                    prod.find_lowest_energy_conformer(hmethod=hmethod)
                    prod.optimise(method=hmethod)

            rxn.locate_transition_state()
            rxn.find_lowest_energy_ts_conformer()
            rxn.calculate_single_points()

            return None

        for i, reaction in enumerate(self.reactions):
            reaction.name = f"{self.name}_step{i}"
            calculate(idx=i)

        self._balance()
        plot_reaction_profile(self.reactions, units=units, name=self.name)
        return None

    @property
    def n_steps(self) -> int:
        """Number of steps in this multistep reaction"""
        return len(self.reactions)

    def _balance(self) -> None:
        """
        Balance reactants and products in each reaction such that the total
        number of atoms is conserved throughout the reaction. Allows for
        multistep reactions where molecules are added e.g. catalytic cycles
        """

        for i in range(self.n_steps - 1):
            j = i + 1  # Index of the next step in the sequence of reactions

            if self.reactions[j].has_identical_composition_as(
                self.reactions[i]
            ):
                continue

            if self._adds_molecule(i, j):
                self._add_mol_to_reacs_prods(
                    mol=self._added_molecule(i, j), to_idx=j
                )

            elif self._adds_molecule(j, i):
                self._add_mol_to_reacs_prods(
                    mol=self._added_molecule(j, i), from_idx=j
                )

            else:
                raise RuntimeError(
                    "Failed to balance the multistep reaction. "
                    "No reactants present as products of a "
                    "previous step."
                )

        return None

    def _adds_molecule(self, step_idx: int, next_step_idx: int) -> bool:
        """Does reaction 2 have more molecules(atoms) than reaction 1"""

        def total_n_atoms(rxn):
            return sum(m.n_atoms for m in rxn.reacs)

        return total_n_atoms(self.reactions[step_idx]) < total_n_atoms(
            self.reactions[next_step_idx]
        )

    def _add_mol_to_reacs_prods(self, mol, from_idx=0, to_idx=None):
        """Add a molecule to both reactants and products"""
        if to_idx is None:
            to_idx = self.n_steps

        for reaction in self.reactions[from_idx:to_idx]:
            reaction.reacs.append(mol.to_reactant())
            reaction.prods.append(mol.to_product())

        return None

    def _added_molecule(self, step_idx: int, next_step_idx: int) -> "Molecule":
        r"""
        Extract the added molecule going from a step to the next. For example::

            A + B -> C
            C + D -> E

        the added molecule would be D. The balanced sequence is then:
        A + B + D -> C + D -> E
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
            if any(p.has_identical_composition_as(mol) for p in prods):
                continue

            return mol

        raise RuntimeError("Failed to find the added molecule")

    def _set_reactants_from_previous_products(self, step_idx: int) -> None:
        """
        Given a reaction step use the previous reactants

        -----------------------------------------------------------------------
        Arguments:
            step_idx: Index of the step (indexed from 0)

        Raises:
            (ValueError): For invalid arguments

            (RuntimeError): If setting is impossible because the multistep
                            reaction is not sequential
        """
        logger.info("Setting reactants from previous steps products")

        if step_idx == 0:
            raise ValueError(
                "Cannot set the products of step -1 as the "
                "reactants of step 0. No step -1!"
            )

        prev_reaction = self.reactions[step_idx - 1]

        for reactant in self.reactions[step_idx].reacs:
            try:
                matching_prod = next(
                    p
                    for p in prev_reaction.prods
                    if reactant.has_identical_composition_as(p)
                )
            except StopIteration:
                raise RuntimeError(
                    "Failed to find a matching product for "
                    f"{reactant} in {prev_reaction}"
                )

            reactant.atoms = matching_prod.atoms.copy()
            reactant.conformers = matching_prod.conformers.copy()
            reactant.energies = matching_prod.energies.copy()

        return None

    @staticmethod
    def _checked_are_reactions(args):
        """Check that all objects in a list are autodE Reaction instances"""

        if not all(isinstance(item, Reaction) for item in args):
            raise ValueError(
                "Cannot initialise a multistep reaction from: "
                f"{args}. Must all be autode Reaction instances"
            )

        return args
