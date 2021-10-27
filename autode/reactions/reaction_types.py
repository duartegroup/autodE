from autode.exceptions import ReactionFormationFailed
from autode.log import logger


class ReactionType:

    def __eq__(self, other):
        return self.name == other.name

    def __init__(self, name):

        self.name = name


Addition = ReactionType(name='addition')
Dissociation = ReactionType(name='dissociation')
Substitution = ReactionType(name='substitution')
Elimination = ReactionType(name='elimination')
Rearrangement = ReactionType(name='rearrangement')


def classify(reactants, products):

    if len(reactants) == 2 and len(products) == 1:
        logger.info('Classifying reaction as addition')
        return Addition

    elif len(reactants) == 1 and len(products) in [2, 3]:
        logger.info('Classifying reaction as dissociation')
        return Dissociation

    elif len(reactants) == 2 and len(products) == 2:
        logger.info('Classifying reaction as substitution')
        return Substitution

    elif len(reactants) == 2 and len(products) == 3:
        logger.info('Classifying reaction as elimination')
        return Elimination

    elif len(reactants) == 1 and len(products) == 1:
        logger.info('Classifying reaction as rearrangement')
        return Rearrangement

    elif len(reactants) == 0:
        logger.critical('Reaction had no reactants – cannot form a reaction')
        raise ReactionFormationFailed

    elif len(products) == 0:
        logger.critical('Reaction had no products – cannot form a reaction')
        raise ReactionFormationFailed

    else:
        logger.critical('Unsupported reaction type')
        raise NotImplementedError
