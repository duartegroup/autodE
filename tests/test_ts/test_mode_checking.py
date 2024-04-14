import os
import numpy as np

from autode.species.molecule import Molecule
from autode.calculations import Calculation
from autode.transition_states.base import TSbase
from autode.transition_states.base import imag_mode_generates_other_bonds
from autode.transition_states.base import displaced_species_along_mode
from autode.species.molecule import Reactant
from autode.input_output import xyz_file_to_atoms
from autode.atoms import Atom
from autode.bond_rearrangement import BondRearrangement
from autode.methods import ORCA
from .. import testutils

here = os.path.dirname(os.path.abspath(__file__))
orca = ORCA()


@testutils.work_in_zipped_dir(os.path.join(here, "data", "mode_checking.zip"))
def test_imag_modes():
    assert not has_correct_mode(
        "incorrect_ts_mode", bbonds=[(1, 6)], fbonds=[(2, 6)]
    )

    assert has_correct_mode(
        "correct_ts_mode", bbonds=[(0, 1), (1, 2)], fbonds=[(0, 2)]
    )

    assert not has_correct_mode(
        "incorrect_ts_mode_2", bbonds=[(3, 8), (3, 2)], fbonds=[(2, 8)]
    )

    assert has_correct_mode(
        "h_shift_correct_ts_mode", bbonds=[(1, 10)], fbonds=[(5, 10)]
    )

    assert has_correct_mode(
        "ene_hess", bbonds=[(0, 5), (2, 1)], fbonds=[(4, 5)]
    )

    assert has_correct_mode(
        "curtius", bbonds=[(0, 1), (2, 3)], fbonds=[(0, 2)]
    )


@testutils.work_in_zipped_dir(os.path.join(here, "data", "mode_checking.zip"))
def test_graph_no_other_bonds():
    ts = TSbase(
        atoms=xyz_file_to_atoms("h_shift_correct_ts_mode.xyz"),
        mult=2,
        bond_rearr=BondRearrangement(
            breaking_bonds=[(1, 10)], forming_bonds=[(5, 10)]
        ),
    )

    calc = Calculation(
        name="h_shift",
        molecule=ts,
        method=orca,
        keywords=orca.keywords.opt_ts,
        n_cores=1,
    )
    calc.set_output_filename("h_shift_correct_ts_mode.out")

    assert ts.hessian is not None

    f_ts = displaced_species_along_mode(ts, mode_number=6, disp_factor=1.0)
    b_ts = displaced_species_along_mode(ts, mode_number=6, disp_factor=-1.0)

    assert not imag_mode_generates_other_bonds(
        ts=ts, f_species=f_ts, b_species=b_ts
    )


def test_disp_molecule_has_same_solvent():
    mol = Molecule(smiles="[H][H]", solvent_name="water")
    mol.hessian = np.eye(3 * mol.n_atoms, 3 * mol.n_atoms)

    disp_mol = displaced_species_along_mode(
        species=mol,
        mode_number=1,
    )
    assert disp_mol.solvent is not None
    assert disp_mol.solvent == mol.solvent


def has_correct_mode(name, fbonds, bbonds):
    calc = Calculation(
        name=name,
        molecule=Reactant(atoms=[Atom("H")], mult=2),
        method=orca,
        keywords=orca.keywords.opt_ts,
        n_cores=1,
    )
    # need to bypass the pre-calculation checks on the molecule. e.g. valid spin state
    calc.molecule = reactant = Reactant(
        name="r", atoms=xyz_file_to_atoms(f"{name}.xyz")
    )

    calc.set_output_filename(f"{name}.out")

    # Don't require all bonds to be breaking/making in a 'could be ts' function
    ts = TSbase(
        atoms=reactant.atoms,
        bond_rearr=BondRearrangement(
            breaking_bonds=bbonds, forming_bonds=fbonds
        ),
    )
    ts.hessian = reactant.hessian

    return ts.imag_mode_has_correct_displacement(req_all=False)
