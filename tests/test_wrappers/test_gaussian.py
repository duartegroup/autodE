import pytest
import os
import numpy as np
from autode.wrappers.G09 import (
    G09,
    _print_custom_basis,
    _get_keywords,
    _n_ecp_elements,
    _add_opt_option,
)
from autode.wrappers.G16 import G16
from autode.calculations import Calculation, CalculationInput
from autode.constraints import Constraints
from autode.species.molecule import Molecule
from autode.wrappers import keywords as kwds
from autode.values import PotentialEnergy
from autode.wrappers.keywords.basis_sets import def2tzecp, def2tzvp
from autode.wrappers.keywords.functionals import pbe0
from autode.wrappers.keywords.keywords import OptKeywords, SinglePointKeywords
from autode.exceptions import NoInputError, CalculationException
from autode.point_charges import PointCharge
from autode.atoms import Atom
from .. import testutils

here = os.path.dirname(os.path.abspath(__file__))
test_mol = Molecule(name="methane", smiles="C")
method = G09()

opt_keywords = OptKeywords(["PBE1PBE/Def2SVP", "Opt"])
optts_keywords = OptKeywords(
    [
        "PBE1PBE/Def2SVP",
        "Freq",
        "Opt=(TS, CalcFC, NoEigenTest, "
        "MaxCycles=100, MaxStep=10, NoTrustUpdate)",
    ]
)

sp_keywords = SinglePointKeywords(["PBE1PBE/Def2SVP"])


def test_printing_ecp():

    tmp_file = open("tmp.com", "w")
    tmp_mol = Molecule(smiles="[H][Pd][H]")
    tmp_mol.constraints = Constraints(distance={}, cartesian=[])

    keywords = kwds.OptKeywords(keyword_list=[def2tzecp])
    assert _n_ecp_elements(keywords, molecule=tmp_mol) == 1
    # Light elements should not default to ECPs
    assert _n_ecp_elements(keywords, molecule=Molecule(smiles="O")) == 0
    # no ECP keywords -> no elements needing an ECP
    assert (
        _n_ecp_elements(kwds.OptKeywords(keyword_list=[]), molecule=tmp_mol)
        == 0
    )

    calc_input = CalculationInput(
        keywords, added_internals=None, point_charges=None
    )

    with pytest.raises(RuntimeError):
        _print_custom_basis(tmp_file, molecule=tmp_mol, calc_input=calc_input)

    calc_input.keywords = kwds.OptKeywords(
        keyword_list=[pbe0, def2tzvp, def2tzecp]
    )
    _print_custom_basis(tmp_file, molecule=tmp_mol, calc_input=calc_input)
    assert os.path.exists("basis.gbs")

    os.remove("tmp.com")
    os.remove("basis.gbs")

    # Should have GenECP in the keywords rather than the ECP or basis
    # definitions
    g09_kwds = _get_keywords(calc_input, molecule=tmp_mol)
    assert not any(kwd.lower() == "def2tzvp" for kwd in g09_kwds)


def test_add_opt_option():

    keywds = ["Opt=Loose"]
    _add_opt_option(keywds, "MaxCycles=10")
    assert keywds[0].lower() == "opt=(loose, maxcycles=10)"


def test_input_print_max_opt():

    keywds = opt_keywords.copy()
    keywds.max_opt_cycles = 10

    str_keywords = _get_keywords(CalculationInput(keywds), molecule=test_mol)

    # Should be only a single instance of the maxcycles declaration
    assert sum("maxcycles=10" in kw.lower() for kw in str_keywords) == 1


@testutils.work_in_zipped_dir(os.path.join(here, "data", "g09.zip"))
def test_get_gradients():

    ester = Molecule(
        name="ester",
        atoms=[
            Atom("C", -1.82707, 0.08502, 0.12799),
            Atom("C", -0.42971, 0.07495, -0.39721),
            Atom("O", 0.47416, -0.05624, 0.58034),
            Atom("C", 1.84921, -0.11372, 0.17588),
            Atom("O", -0.11743, 0.16179, -1.57499),
            Atom("H", -1.93163, 0.83516, 0.91779),
            Atom("H", -2.52831, 0.29353, -0.68072),
            Atom("H", -2.05617, -0.89219, 0.56728),
            Atom("H", 2.41930, -0.21232, 1.09922),
            Atom("H", 2.13056, 0.80382, -0.3467),
            Atom("H", 2.01729, -0.97969, -0.4689),
        ],
    )

    calc = Calculation(
        name="ester",
        molecule=ester,
        method=method,
        keywords=method.keywords.opt,
    )
    calc.set_output_filename("ester_opt_g09.log")

    gradients = ester.gradient
    assert gradients is not None
    assert gradients.shape == (ester.n_atoms, 3)


@testutils.work_in_zipped_dir(os.path.join(here, "data", "g09.zip"))
def test_gauss_opt_calc():

    methylchloride = Molecule(
        name="CH3Cl", smiles="[H]C([H])(Cl)[H]", solvent_name="water"
    )
    calc = Calculation(
        name="opt",
        molecule=methylchloride,
        method=method,
        keywords=opt_keywords,
    )
    calc.run()

    assert os.path.exists("opt_g09.com")
    assert os.path.exists("opt_g09.log")
    assert len(calc.get_final_atoms()) == 5
    assert calc.get_energy() == -499.729222331
    assert calc.output.exists
    assert calc.output.file_lines is not None
    assert methylchloride.imaginary_frequencies is None

    assert calc.input.filename == "opt_g09.com"
    assert calc.output.filename == "opt_g09.log"
    assert calc.terminated_normally
    assert calc.optimisation_converged()
    assert calc.optimisation_nearly_converged() is False

    charges = calc.get_atomic_charges()
    assert len(charges) == methylchloride.n_atoms

    # Should be no very large atomic charges in this molecule
    assert all(-1.0 < c < 1.0 for c in charges)

    gradients = calc.get_gradients()
    assert len(gradients) == methylchloride.n_atoms
    assert len(gradients[0]) == 3

    # Should be no large forces for an optimised molecule
    assert sum(gradients[0]) < 0.1

    # Should have a small non-zero last energy change
    print(calc.optimiser.last_energy_change)
    assert calc.optimiser.last_energy_change == PotentialEnergy(1.127e-5, "Ha")


@testutils.work_in_zipped_dir(os.path.join(here, "data", "g09.zip"))
def test_gauss_optts_calc():

    test_mol = Molecule(name="methane", smiles="C")
    test_mol.graph.add_active_edge(0, 1)

    calc = Calculation(
        name="test_ts_reopt_optts",
        molecule=test_mol,
        method=method,
        keywords=optts_keywords,
    )
    calc.run()
    assert calc.output.exists

    assert os.path.exists("test_ts_reopt_optts_g09.com")

    bond_added = False
    for line in open("test_ts_reopt_optts_g09.com", "r"):
        if "B" in line and len(line.split()) == 3:
            bond_added = True
            assert line.split()[0] == "B"
            assert line.split()[1] == "1"
            assert line.split()[2] == "2"

    assert bond_added

    mol = Molecule(atoms=calc.get_final_atoms())
    mol.calc_thermo(calc=calc, ss="1atm", lfm_method="igm")

    assert calc.terminated_normally
    assert calc.optimisation_converged()
    assert calc.optimisation_nearly_converged() is False
    assert len(mol.imaginary_frequencies) == 1

    assert -40.324 < mol.free_energy < -40.322
    assert -40.301 < mol.enthalpy < -40.298


def test_bad_gauss_output():

    calc = Calculation(
        name="no_output",
        molecule=test_mol,
        method=method,
        keywords=opt_keywords,
    )
    calc.output_file_lines = []
    calc.rev_output_file_lines = []

    with pytest.raises(CalculationException):
        _ = calc.get_energy()

    with pytest.raises(CalculationException):
        calc.get_final_atoms()


@testutils.work_in_zipped_dir(os.path.join(here, "data", "g09.zip"))
def test_fix_angle_error():

    os.chdir(os.path.join(here, "data", "g09"))

    mol = Molecule(smiles="CC/C=C/CO")
    mol.name = "molecule"

    calc = Calculation(
        name="angle_fail", molecule=mol, method=method, keywords=opt_keywords
    )
    calc.run()

    assert os.path.exists("angle_fail_g09_cartesian.com") is True
    assert os.path.exists("angle_fail_g09_internal.com") is True
    assert calc.output.filename == "angle_fail_g09_internal.log"
    assert calc.terminated_normally


@testutils.work_in_zipped_dir(os.path.join(here, "data", "g09.zip"))
def test_constraints():

    a = test_mol.copy()
    a.constraints.distance = {(0, 1): 1.2}
    calc = Calculation(
        name="const_dist_opt", molecule=a, method=method, keywords=opt_keywords
    )
    calc.run()
    opt_atoms = calc.get_final_atoms()

    assert (
        1.199 < np.linalg.norm(opt_atoms[0].coord - opt_atoms[1].coord) < 1.201
    )

    b = test_mol.copy()
    b.constraints.cartesian = [0]
    calc = Calculation(
        name="const_cart_opt", molecule=b, method=method, keywords=opt_keywords
    )
    calc.run()
    opt_atoms = calc.get_final_atoms()
    assert np.linalg.norm(test_mol.atoms[0].coord - opt_atoms[0].coord) < 1e-3


@testutils.work_in_zipped_dir(os.path.join(here, "data", "g09.zip"))
def test_single_atom_opt():

    mol = Molecule(smiles="[H]")
    mol.name = "molecule"

    calc = Calculation(
        name="H", molecule=mol, method=method, keywords=opt_keywords, n_cores=2
    )
    calc.generate_input()
    assert os.path.exists("H_g09.com")

    input_file_lines = open("H_g09.com", "r").readlines()

    n_cores_set = False
    for line in input_file_lines:
        if "PBE" in line:
            assert "Opt" not in line
        if "%nprocshared=2" in line:
            n_cores_set = True

    assert n_cores_set


@testutils.work_in_zipped_dir(os.path.join(here, "data", "g09.zip"))
def test_point_charge_calc():
    # Methane single point using a point charge with a unit positive charge
    # located at (10, 10, 10)

    calc = Calculation(
        name="methane_point_charge",
        molecule=test_mol,
        method=method,
        keywords=sp_keywords,
        point_charges=[PointCharge(charge=1.0, x=10.0, y=10.0, z=10.0)],
    )
    calc.run()

    # Assert that the input file is in the expected configuration
    for line in open("methane_point_charge_g09.com", "r"):
        if "PBE" in line:
            assert "Charge" in line

        if len(line.split()) == 4:
            if not line.split()[0].isdigit():
                continue

            x, y, z, charge = line.split()
            assert float(x) == 10.0
            assert float(y) == 10.0
            assert float(z) == 10.0
            assert float(charge) == 1.0

    assert -40.428 < calc.get_energy() < -40.427

    # Gaussian needs x-matrix and nosymm in the input line to run optimisations
    # with point charges..
    for opt_keyword in ["Opt", "Opt=Tight", "Opt=(Tight)"]:
        calc = Calculation(
            name="methane_point_charge_o",
            molecule=test_mol,
            method=method,
            keywords=OptKeywords(["PBE1PBE/Def2SVP", opt_keyword]),
            point_charges=[PointCharge(charge=1.0, x=3.0, y=3.0, z=3.0)],
        )
        calc.generate_input()

        for line in open("methane_point_charge_o_g09.com", "r").readlines():
            if "PBE" in line:
                assert "charge" in line.lower()
                assert "z-matrix" in line.lower() and "nosymm" in line.lower()
                break


@testutils.work_in_zipped_dir(os.path.join(here, "data", "g09.zip"))
def test_external_basis_set_file():

    """

    Example calculation with a custom basis set and ECP
    -----------------------------------------------
    # Opt M062X EmpiricalDispersion=GD3BJ genecp

     name of calc

    0 1
    xyz coordinates

    @bs1.gbs

    """

    # This test needs to not change the filename based on the input as the
    # keywords depend on the current working directory, thus is not generally
    # going to be the same
    if os.getenv("AUTODE_FIXUNIQUE", True) != "False":
        return

    custom = G16()

    basis_path = os.path.join(os.getcwd(), "bs1.gbs")
    custom.keywords.set_opt_basis_set(basis_path)
    assert custom.keywords.opt.basis_set.has_only_name

    custom.keywords.sp.basis = basis_path

    pd_cl2 = Molecule("pd_cl2.xyz")
    pd_cl2.single_point(method=custom)
    assert pd_cl2.energy is not None
    # ensure the energy is in the right ball-park
    assert np.abs(pd_cl2.energy - -1046.7287) < 1e-2


@testutils.work_in_zipped_dir(os.path.join(here, "data", "g09.zip"))
def test_xtb_optts():

    g09 = G09()

    kwd_list = [
        "External='xtb-gaussian'",
        "Opt(TS, CalcFC, NoEigenTest, MaxCycles=100, MaxStep=10, "
        "NoTrustUpdate, NoMicro)",
        "IOp(3/5=30)",
    ]

    orca_ts = Molecule(
        atoms=[
            Atom("F", -5.15221, 4.39259, 0.10105),
            Atom("Cl", -1.03103, 4.55239, -0.06066),
            Atom("C", -3.15949, 4.47211, 0.02185),
            Atom("H", -3.27697, 3.86557, -0.86787),
            Atom("H", -3.20778, 3.99594, 0.99353),
            Atom("H", -3.30829, 5.54289, -0.04740),
        ],
        charge=-1,
        solvent_name="water",
    )

    calc = Calculation(
        name="tmp",
        molecule=orca_ts,
        method=g09,
        keywords=OptKeywords(kwd_list),
    )
    calc.run()

    # Even though a Hessian is not requested it should be added
    assert calc.get_hessian() is not None
    assert np.isclose(calc.get_energy().to("Ha"), -13.1297380, atol=1e-5)
