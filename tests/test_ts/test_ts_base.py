import pytest

from autode.config import Config
from autode.atoms import Atom
from autode.values import Frequency
from autode.transition_states.ts_guess import TSguess
from autode.species.molecule import Reactant, Product
from autode.bond_rearrangement import BondRearrangement


def h3_ts_guess():
    return TSguess(atoms=[Atom("H"), Atom("H", x=0.7), Atom("H", 1.4)], mult=2)


def test_invalid_init_different_reactants_products():

    with pytest.raises(ValueError):
        _ = TSguess(
            atoms=[Atom("H")],
            reactant=Reactant(atoms=[Atom("H")], solvent_name="water"),
            product=Product(atoms=[Atom("H")], solvent_name="dcm"),
        )


def test_invalid_init_different_reactant():

    with pytest.raises(ValueError):
        _ = TSguess(
            atoms=[Atom("H")],
            reactant=Reactant(atoms=[Atom("H")], solvent_name="water"),
            solvent_name="dcm",
        )


def test_mode_checking_without_bond_rearr():

    ts_guess = h3_ts_guess()

    with pytest.raises(Exception):
        _ = ts_guess.could_have_correct_imag_mode


def test_could_have_correct_imag_mode():
    class TmpHess:
        def __init__(self, freqs: list):
            self.frequencies_proj = freqs

    ts_guess = h3_ts_guess()
    ts_guess.bond_rearrangement = BondRearrangement(
        forming_bonds=[(0, 1)], breaking_bonds=[(1, 2)]
    )

    ts_guess._hess = TmpHess(freqs=[])  # No frequencies
    assert ts_guess.imaginary_frequencies is None
    assert not ts_guess.could_have_correct_imag_mode

    ts_guess._hess = TmpHess(freqs=[Config.min_imag_freq / 2, Frequency(1000)])
    assert not ts_guess.could_have_correct_imag_mode
