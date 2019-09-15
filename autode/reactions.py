from autode.log import logger


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


def classify(reacs, prods):

    if len(reacs) == 2 and len(prods) == 1:
        logger.info('Classifying reaction as addition')
        return Addition

    elif len(reacs) == 1 and len(prods) == 2:
        logger.info('Classifying reaction as dissociation')
        return Dissociation

    elif len(reacs) == 2 and len(prods) == 2:
        logger.info('Classifying reaction as substitution')
        return Substitution

    elif len(reacs) == 2 and len(prods) == 3:
        logger.info('Classifying reaction as elimination')
        return Elimination

    elif len(reacs) == 1 and len(prods) == 1:
        logger.info('Classifying reaction as rearrangement')
        return Rearrangement

    else:
        logger.critical('Unsupported reaction type')
        exit()
