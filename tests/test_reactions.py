from autode.reactions.reaction_types import classify
from autode.exceptions import ReactionFormationFalied
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

    with pytest.raises(ReactionFormationFalied):
        classify([], [])
