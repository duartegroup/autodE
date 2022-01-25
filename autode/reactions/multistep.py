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

        if not all(isinstance(rxn, Reaction) for rxn in args):
            raise ValueError('Cannot initialise a multistep reaction from: '
                             f'{args}. Must all be autode Reaction instances')

        self.reactions = args

    def calculate_reaction_profile(self,
                                   units: Union['autode.units.Unit', str] = 'kcal mol-1',
                                   ) -> None:
        """
        Calculate a multistep reaction profile using the products of step 1
        as the reactants of step 2 etc. Example

        .. code-block::

            >>> import autode as ade
            >>> step1 = ade.Reaction('')
            >>> step2 = ade.Reaction('')
            >>> rxn = ade.MultiStepReaction(step1, step2)
            >>> rxn.calculate_reaction_profile()
        """
        logger.info('Calculating reaction profile')
        h_method = get_hmethod() if Config.hmethod_conformers else None

        @work_in(self.name)
        def calculate(reaction, prev_products):

            for mol in reaction.reacs + reaction.prods:

                # If this molecule is one of the previous products don't
                # regenerate conformers
                skip_conformers = False
                for prod in prev_products:

                    # Has the same atoms & charge etc. as in the unique string
                    if str(mol) == str(prod):
                        mol = prod
                        skip_conformers = True

                if not skip_conformers:
                    mol.find_lowest_energy_conformer(hmethod=h_method)

            reaction.optimise_reacs_prods()
            reaction.locate_transition_state()
            reaction.find_lowest_energy_ts_conformer()
            reaction.calculate_single_points()

            return None

        # For all the reactions calculate the reactants, products and TS
        for i, r in enumerate(self.reactions):
            r.name = f'{self.name}_step{i}'

            if i == 0:
                # First reaction has no previous products
                calculate(reaction=r, prev_products=[])

            else:
                calculate(reaction=r, prev_products=self.reactions[i-1].prods)

        plot_reaction_profile(self.reactions, units=units, name=self.name)
        return None
