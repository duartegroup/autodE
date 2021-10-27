from autode.reactions.reaction_types import classify, Addition
from autode.exceptions import ReactionFormationFailed
import pytest


def test_classify():
    # Can do this with integers are in classify there is no check on their
    # contents, only their length

    addition = classify([0, 0], [0])
    assert addition.name == 'addition'

    dissociation = classify([0], [0, 0])
    assert dissociation.name == 'dissociation'

    substitution = classify([0, 0], [0, 0])
    assert substitution.name == 'substitution'

    elimination = classify([0, 0], [0, 0, 0])
    assert elimination.name == 'elimination'

    rearrangement = classify([0], [0])
    assert rearrangement.name == 'rearrangement'

    # Needs to have at least some reactants and products
    with pytest.raises(ReactionFormationFailed):
        _ = classify([], [])

    with pytest.raises(ReactionFormationFailed):
        _ = classify([0], [])

    with pytest.raises(ReactionFormationFailed):
        _ = classify([], [0])

    # 3 -> 3 reactions are not currently supported
    with pytest.raises(NotImplementedError):
        _ = classify([0, 1, 2], [3, 4, 5])


def test_equality():

    assert Addition == Addition
