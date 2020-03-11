from autode.log import logger
from autode.exceptions import ReactionFormationFalied


class Addition:
    name = 'addition'


class Dissociation:
    name = 'dissociation'


class Substitution:
    name = 'substitution'


class Elimination:
    name = 'elimination'


class Rearrangement:
    name = 'rearrangement'


def classify(reactants, products):

    if len(reactants) == 2 and len(products) == 1:
        logger.info('Classifying reaction as addition')
        return Addition

    elif len(reactants) == 1 and len(products) == 2:
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
        raise ReactionFormationFalied

    elif len(products) == 0:
        logger.critical('Reaction had no products – cannot form a reaction')
        raise ReactionFormationFalied

    else:
        logger.critical('Unsupported reaction type')
        raise NotImplementedError
