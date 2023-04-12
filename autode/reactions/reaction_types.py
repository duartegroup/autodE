from typing import Sequence, Optional, TYPE_CHECKING

from autode.exceptions import ReactionFormationFailed
from autode.log import logger

if TYPE_CHECKING:
    from autode.species import Species


class ReactionType:
    """Type of a reaction e.g. Addition"""

    def __init__(self, name: str):
        """
        Reaction type with only a name

        -----------------------------------------------------------------------
        Arguments:
            name: Name of the reaction type
        """
        self.name = name

    def __eq__(self, other) -> bool:
        """Equality of two types is just based on their names"""
        return isinstance(other, ReactionType) and self.name == other.name


Addition = ReactionType(name="addition")
Dissociation = ReactionType(name="dissociation")
Substitution = ReactionType(name="substitution")
Elimination = ReactionType(name="elimination")
Rearrangement = ReactionType(name="rearrangement")


def classify(
    reactants: Sequence["Species"],
    products: Sequence["Species"],
) -> Optional[ReactionType]:
    """
    Classify a reaction into a type given some reactants and products

    ---------------------------------------------------------------------------
    Arguments:
        reactants:
        products:

    Returns:
        (autode.reactions.reaction_types.ReactionType): Reaction type
    """
    n_reactants, n_products = len(reactants), len(products)

    if n_reactants == n_products == 0:
        return None

    if n_reactants == 0:
        raise ReactionFormationFailed(
            f"Reaction had 0 reactants and "
            f"{n_products} products. A reaction "
            f"requires at least 1 reactant!"
        )

    if n_products == 0:
        raise ReactionFormationFailed(
            f"Reaction had 0 products but "
            f"{n_reactants} reactants. A reaction "
            f"requires at least 1 product!"
        )

    if n_reactants == 2 and n_products == 1:
        logger.info("Classifying reaction as addition")
        return Addition

    elif n_reactants == 1 and n_products in [2, 3]:
        logger.info("Classifying reaction as dissociation")
        return Dissociation

    elif n_reactants == 2 and n_products == 2:
        logger.info("Classifying reaction as substitution")
        return Substitution

    elif n_reactants == 2 and n_products == 3:
        logger.info("Classifying reaction as elimination")
        return Elimination

    elif n_reactants == 1 and n_products == 1:
        logger.info("Classifying reaction as rearrangement")
        return Rearrangement

    else:
        raise NotImplementedError(
            "Unsupported reaction type: " f"{n_reactants} -> {n_products}"
        )
