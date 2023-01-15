from autode.wrappers.MOPAC import MOPAC
from autode.wrappers.MOPAC import get_keywords, _get_atoms_linear_interp
from autode.exceptions import (
    CouldNotGetProperty,
    UnsupportedCalculationInput,
    CalculationException,
)
from autode.calculations import Calculation, CalculationInput
from autode.species.molecule import Molecule
from autode.solvent import ImplicitSolvent
from autode.atoms import Atom
from autode.constants import Constants
from autode.config import Config
from autode.values import PotentialEnergy
from autode.point_charges import PointCharge
from .. import testutils
import numpy as np
import os
import pytest

here = os.path.dirname(os.path.abspath(__file__))
method = MOPAC()


def mecl():
    return Molecule(
        name="CH3Cl", smiles="[H]C([H])(Cl)[H]", solvent_name="water"
    )


@testutils.work_in_zipped_dir(os.path.join(here, "data", "mopac.zip"))
def test_mopac_opt_calculation():

    mol = mecl()
    calc = Calculation(
        name="opt",
        molecule=mol,
        method=method,
        keywords=Config.MOPAC.keywords.opt,
    )
    calc.run()

    assert os.path.exists("opt_mopac.mop") is True
    assert os.path.exists("opt_mopac.out") is True
    assert len(calc.get_final_atoms()) == 5

    # Actual energy in Hartrees
    energy = Constants.eV_to_ha * -430.43191
    assert energy - 0.0001 < calc.get_energy() < energy + 0.0001

    assert calc.output.exists
    assert calc.output.file_lines is not None
    assert calc.input.filename == "opt_mopac.mop"
    assert calc.output.filename == "opt_mopac.out"
    assert calc.terminated_normally
    assert calc.optimiser.converged
    assert mol.gradient is None


@testutils.work_in_zipped_dir(os.path.join(here, "data", "mopac.zip"))
def test_mopac_with_pc():

    calc = Calculation(
        name="opt_pc",
        molecule=mecl(),
        method=method,
        keywords=Config.MOPAC.keywords.opt,
        point_charges=[PointCharge(1, x=4, y=4, z=4)],
    )
    calc.run()

    assert os.path.exists("opt_pc_mopac.mop") is True
    assert os.path.exists("opt_pc_mopac.out") is True
    assert len(calc.get_final_atoms()) == 5

    # Actual energy in Hartrees without any point charges
    energy = Constants.eV_to_ha * -430.43191
    assert np.abs(calc.get_energy() - energy) > 0.0001


@testutils.work_in_zipped_dir(os.path.join(here, "data", "mopac.zip"))
def test_other_spin_states():

    o_singlet = Molecule(atoms=[Atom("O")], mult=1)
    o_singlet.name = "molecule"

    calc = Calculation(
        name="O_singlet",
        molecule=o_singlet,
        method=method,
        keywords=Config.MOPAC.keywords.sp,
    )
    calc.run()
    singlet_energy = calc.get_energy()

    o_triplet = Molecule(atoms=[Atom("O")], mult=3)
    o_triplet.name = "molecule"

    calc = Calculation(
        name="O_triplet",
        molecule=o_triplet,
        method=method,
        keywords=Config.MOPAC.keywords.sp,
    )
    calc.run()
    triplet_energy = calc.get_energy()

    assert triplet_energy < singlet_energy

    h_doublet = Molecule(atoms=[Atom("H")], mult=2)
    h_doublet.name = "molecule"

    calc = Calculation(
        name="h",
        molecule=h_doublet,
        method=method,
        keywords=Config.MOPAC.keywords.sp,
    )
    calc.run()

    # Open shell doublet should work
    assert calc.get_energy() is not None

    h_quin = Molecule(atoms=[Atom("H")], mult=5)
    h_quin.name = "molecule"

    with pytest.raises(CalculationException):
        calc = Calculation(
            name="h",
            molecule=h_quin,
            method=method,
            keywords=Config.MOPAC.keywords.sp,
        )
        calc.run()

    os.chdir(here)


@testutils.work_in_zipped_dir(os.path.join(here, "data", "mopac.zip"))
def test_bad_geometry():

    # Calculation with the wrong spin state should fail
    calc = Calculation(
        name="h2_overlap_opt",
        molecule=Molecule(atoms=[Atom("H"), Atom("H")]),
        method=method,
        keywords=Config.MOPAC.keywords.opt,
    )

    calc.output.filename = "h2_overlap_opt_mopac.out"
    assert not calc.terminated_normally

    with pytest.raises(Exception):
        _ = calc.get_energy()

    assert not method.optimiser_from(calc).converged


@testutils.work_in_zipped_dir(os.path.join(here, "data", "mopac.zip"))
def test_constrained_opt():

    methane = Molecule(name="methane", smiles="C")

    calc = Calculation(
        name="methane_opt",
        molecule=methane,
        method=method,
        keywords=Config.MOPAC.keywords.opt,
    )
    calc.run()
    opt_energy = calc.get_energy()

    # Constrained optimisation with a C–H distance of 1.2 Å
    # (carbon is the first atom in the file)
    methane.constraints.distance = {(0, 1): 1.2}
    const = Calculation(
        name="methane_const",
        molecule=methane,
        method=method,
        keywords=Config.MOPAC.keywords.opt,
    )
    const.run()

    assert opt_energy < const.get_energy()
    assert calc.get_hessian() is None


@testutils.work_in_zipped_dir(os.path.join(here, "data", "mopac.zip"))
def test_grad():

    h2 = Molecule(name="h2", atoms=[Atom("H"), Atom("H", x=0.5)])

    grad_calc = Calculation(
        name="h2_grad",
        molecule=h2,
        method=method,
        keywords=Config.MOPAC.keywords.grad,
    )
    grad_calc.run()
    energy = grad_calc.get_energy()
    assert energy is not None

    gradients = grad_calc.get_gradients()
    assert gradients.shape == (2, 3)

    delta_r = 1e-5
    h2_disp = Molecule(
        name="h2_disp", atoms=[Atom("H"), Atom("H", x=0.5 + delta_r)]
    )
    h2_disp.single_point(method)

    delta_energy = h2_disp.energy - energy  # Ha]
    grad = delta_energy / delta_r  # Ha A^-1

    # Difference between the absolute and finite difference approximation
    assert np.abs(gradients[1, 0] - grad) < 1e-1


@testutils.work_in_zipped_dir(os.path.join(here, "data", "mopac.zip"))
def test_broken_grad():

    h2 = Molecule(name="h2", atoms=[Atom("H"), Atom("H", x=0.5)])
    grad_calc_broken = Calculation(
        name="h2_grad",
        molecule=h2,
        method=method,
        keywords=Config.MOPAC.keywords.grad,
    )
    grad_calc_broken.output.filename = "h2_grad_broken.out"

    with pytest.raises(CouldNotGetProperty):
        _ = grad_calc_broken.get_gradients()


@testutils.work_in_zipped_dir(os.path.join(here, "data", "mopac.zip"))
def test_new_energy():

    h2o = Molecule(smiles="O")
    calc = Calculation(
        name="H2O",
        molecule=h2o,
        method=method,
        keywords=Config.MOPAC.keywords.grad,
    )
    calc.set_output_filename("H2O_mopac_new.out")

    assert h2o.energy == PotentialEnergy(-322.5968, units="eV")


def test_termination_short():

    calc = Calculation(
        name="test",
        molecule=mecl(),
        method=method,
        keywords=Config.MOPAC.keywords.sp,
    )

    calc.output.filename = "test.out"
    with open(calc.output.filename, "w") as test_output:
        print("JOB ENDED NORMALLY", "another line", sep="\n", file=test_output)

    assert calc.terminated_normally
    os.remove(calc.output.filename)


def test_mopac_keywords():

    calc_input = CalculationInput(
        keywords=Config.MOPAC.keywords.sp,
        added_internals=None,
        point_charges=None,
    )

    keywords = get_keywords(calc_input=calc_input, molecule=mecl())
    assert any("1scf" == kw.lower() for kw in keywords)

    calc_input.keywords = Config.MOPAC.keywords.grad
    keywords = get_keywords(calc_input=calc_input, molecule=mecl())
    assert any("grad" == kw.lower() for kw in keywords)

    h = Molecule(name="H", smiles="[H]")
    assert h.mult == 2

    keywords = get_keywords(calc_input=calc_input, molecule=h)
    assert any("doublet" == kw.lower() for kw in keywords)


def test_get_version_no_output():

    calc = Calculation(
        name="test",
        molecule=mecl(),
        method=method,
        keywords=method.keywords.sp,
    )
    calc.output.filename = "test.out"

    with open(calc.output.filename, "w") as test_output:
        print("Error 1", "Error 2", sep="\n", file=test_output)

    assert not calc.terminated_normally
    assert method.version_in(calc) == "???"

    os.remove(calc.output.filename)


def test_mopac_solvent_no_dielectric():

    mol = mecl()
    mol.solvent = ImplicitSolvent("X", smiles="X", aliases=["X"], mopac="X")

    calc = Calculation(
        "tmp", molecule=mol, method=method, keywords=method.keywords.sp
    )

    # Cannot generate an input if the solvent does not have a defined
    # dielectric constant in the dictionary
    with pytest.raises(UnsupportedCalculationInput):
        calc.generate_input()

    if os.path.exists("tmp_mopac.mop"):
        os.remove("tmp_mopac.mop")


def test_shifted_atoms():

    atoms = [Atom("H", 0.0, 0.0, 0.0), Atom("H", 0.0, 0.0, 2.0)]

    new_atoms = _get_atoms_linear_interp(
        atoms, bonds=[(0, 1)], final_distances=[1.0]
    )

    # Linear interpolation of the coordinates should move the atom either
    # end of the bond half way
    assert (
        np.linalg.norm(new_atoms[0].coord - np.array([0.0, 0.0, 0.5])) < 1e-6
    )
    assert (
        np.linalg.norm(new_atoms[1].coord - np.array([0.0, 0.0, 1.5])) < 1e-6
    )
