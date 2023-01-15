import pytest
from autode.reactions.reaction_types import classify, Addition
from autode.exceptions import ReactionFormationFailed


def test_classify():
    """
    Testing with integers, as in classify there is no check on the type
    of reactants and products, only their length
    """

    addition = classify([0, 0], [0])
    assert addition.name == "addition"

    dissociation = classify([0], [0, 0])
    assert dissociation.name == "dissociation"

    substitution = classify([0, 0], [0, 0])
    assert substitution.name == "substitution"

    elimination = classify([0, 0], [0, 0, 0])
    assert elimination.name == "elimination"

    rearrangement = classify([0], [0])
    assert rearrangement.name == "rearrangement"

    assert classify([], []) is None

    # Needs to have at least some reactants or products
    with pytest.raises(ReactionFormationFailed):
        _ = classify([0], [])

    with pytest.raises(ReactionFormationFailed):
        _ = classify([], [0])

    # 3 -> 3 reactions are not currently supported
    with pytest.raises(NotImplementedError):
        _ = classify([0, 1, 2], [3, 4, 5])


def test_equality():

    assert Addition == Addition
    assert Addition != 0
