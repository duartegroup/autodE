from autode.plotting import plot_reaction_profile
from autode.units import KcalMol
from autode.methods import get_hmethod
from autode.config import Config
from autode.reactions.reaction import Reaction
from autode.log import logger
from autode.utils import work_in


class MultiStepReaction:

    def calculate_reaction_profile(self, units=KcalMol):
        """Calculate a multistep reaction profile using the products of step 1
        as the reactants of step 2 etc."""
        logger.info('Calculating reaction profile')
        h_method = get_hmethod()

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
                    mol.find_lowest_energy_conformer(hmethod=h_method if Config.hmethod_conformers else None)

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

    def __init__(self, *args, name='reaction'):
        """
        Reaction with multiple steps

        Arguments:
            *args (autode.reaction.Reaction): Set of reactions to calculate the
                                              reaction profile for
        """
        self.name = str(name)
        self.reactions = args

        assert all(type(reaction) is Reaction for reaction in self.reactions)
