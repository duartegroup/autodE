import os
import numpy as np
from autode import Atom, Reactant, Product, Reaction
from autode.methods import XTB
from autode.bond_rearrangement import BondRearrangement
from autode.neb import NEB
from autode.transition_states.locate_tss import _get_ts_neb_from_adaptive_path
from .. import testutils

here = os.path.dirname(os.path.abspath(__file__))


def _sn2_reaction():

    r0 = Reactant(
        atoms=[
            Atom("C", -0.1087, -0.0058, -0.0015),
            Atom("Cl", 1.6659, 0.0565, -0.0425),
            Atom("H", -0.5267, -0.6146, -0.8284),
            Atom("H", -0.4802, -0.4519, 0.9359),
            Atom("H", -0.5503, 1.0158, -0.0635),
        ]
    )

    r1 = Reactant(atoms=[Atom("F")], charge=-1)
    p0 = Product(
        atoms=[
            Atom("C", -0.0524, -0.0120, 0.0160),
            Atom("F", 1.3238, -0.1464, -0.1423),
            Atom("H", -0.3175, 0.0493, 1.0931),
            Atom("H", -0.3465, 0.9303, -0.4647),
            Atom("H", -0.6073, -0.8212, -0.5021),
        ]
    )
    p1 = Product(atoms=[Atom("Cl")], charge=-1)

    return Reaction(r0, r1, p0, p1, solvent_name="water")


@testutils.work_in_zipped_dir(os.path.join(here, "data", "ts_adapt_neb.zip"))
@testutils.requires_with_working_xtb_install
def test_ts_from_neb_optimised_after_adapt():

    rxn = _sn2_reaction()
    neb = NEB.from_file("dOr2Us_ll_ad_0-5_0-1_path.xyz")
    assert len(neb.images) > 0
    assert neb.images[0].species.solvent is not None

    def get_ts_guess():
        return _get_ts_neb_from_adaptive_path(
            reactant=rxn.reactant,
            product=rxn.product,
            method=XTB(),
            name="dOr2Us_ll_ad_neb_0-5_0-1",
            ad_name="dOr2Us_ll_ad_0-5_0-1",
            bond_rearr=BondRearrangement(
                forming_bonds=[(0, 5)], breaking_bonds=[(0, 1)]
            ),
        )

    ts_guess = get_ts_guess()
    assert ts_guess is not None

    # sum of the H-C-H angles should be close to 360º for the correct TS
    angles = [
        ts_guess.angle(2, 0, 3),
        ts_guess.angle(2, 0, 4),
        ts_guess.angle(3, 0, 4),
    ]
    assert np.isclose(sum([a.to("degrees") for a in angles]), 360, atol=5)

    # if we delete the path then no NEB is possible
    os.remove("dOr2Us_ll_ad_0-5_0-1_path.xyz")
    assert get_ts_guess() is None


@testutils.work_in_zipped_dir(os.path.join(here, "data", "ts_adapt_neb.zip"))
def test_no_ts_guess_without_peak_in_ad_path():

    rxn = _sn2_reaction()
    ts_guess = _get_ts_neb_from_adaptive_path(
        reactant=rxn.reactant,
        product=rxn.product,
        method=XTB(),
        name="dOr2Us_ll_ad_neb_0-5_0-1",
        ad_name="no_peak",
        bond_rearr=BondRearrangement(),
    )
    # No calculations are performed
    assert ts_guess is None
