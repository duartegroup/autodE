import autode.exceptions as ex
from autode.calculations import CalculationInput
from autode.atoms import Atom
from autode.constants import Constants
from autode.calculations import Calculation
from autode.species.molecule import Molecule
from autode.point_charges import PointCharge
from autode.input_output import xyz_file_to_atoms
from autode.wrappers.keywords import (
    SinglePointKeywords,
    OptKeywords,
    HessianKeywords,
)
from autode.wrappers.keywords import Functional, WFMethod, BasisSet
from autode.wrappers.keywords import cpcm
from autode.transition_states.transition_state import TransitionState
from autode.transition_states.ts_guess import TSguess
from autode.wrappers.ORCA import (
    ORCA,
    ORCAOptimiser,
    print_cartesian_constraints,
    print_point_charges,
)
from autode import utils
from .. import testutils
import numpy as np
import pytest

import os

here = os.path.dirname(os.path.abspath(__file__))
test_mol = Molecule(name="methane", smiles="C")
method = ORCA()

sp_keywords = SinglePointKeywords(["PBE", "def2-SVP"])
opt_keywords = OptKeywords(["Opt", "PBE", "def2-SVP"])


@testutils.work_in_zipped_dir(os.path.join(here, "data", "orca.zip"))
def test_orca_opt_calculation():

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

    assert os.path.exists("opt_orca.inp") is True
    assert os.path.exists("opt_orca.out") is True
    assert len(calc.get_final_atoms()) == 5
    assert -499.735 < calc.get_energy() < -499.730
    assert calc.output.exists
    assert calc.output.file_lines is not None
    assert calc.input.filename == "opt_orca.inp"
    assert calc.output.filename == "opt_orca.out"
    assert calc.terminated_normally

    assert calc.optimisation_converged()

    assert calc.optimisation_nearly_converged() is False

    # Should have a partial atomic charge for every atom
    charges = calc.get_atomic_charges()
    assert charges == [-0.006954, -0.147352, 0.052983, 0.052943, 0.053457]

    calc = Calculation(
        name="opt",
        molecule=methylchloride,
        method=method,
        keywords=opt_keywords,
    )

    # If the calculation is not run with calc.run() then there should be no
    # input and the calc should raise that there is no input
    with pytest.raises(ex.NoInputError):
        f = utils.hashable("_execute_external", calc._executor)
        f()


@testutils.work_in_zipped_dir(os.path.join(here, "data", "orca.zip"))
def test_orca_optts_calculation():

    ts = TransitionState.from_species(
        Molecule("test_ts_reopt_optts_orca.xyz", charge=-1)
    )
    ts.graph.add_active_edge(0, 1)

    optts_str = (
        "\n%geom\n"
        "Calc_Hess true\n"
        "Recalc_Hess 40\n"
        "Trust 0.2\n"
        "MaxIter 100\nend"
    )

    calc = Calculation(
        name="test_ts_reopt_optts",
        molecule=ts,
        method=method,
        keywords=opt_keywords + [optts_str],
    )
    calc.run()

    ts.calc_thermo(calc=calc, ss="1atm", sn=1)

    assert os.path.exists("test_ts_reopt_optts_orca.inp")

    assert ts.normal_mode(mode_number=6) is not None
    assert calc.terminated_normally
    assert calc.optimisation_converged()
    assert calc.optimisation_nearly_converged() is False
    assert len(ts.imaginary_frequencies) == 1

    # Gradients should be an n_atom x 3 array
    gradients = calc.get_gradients()
    assert gradients.shape == (ts.n_atoms, 3)

    assert -599.437 < ts.enthalpy < -599.436
    assert -599.469 < ts.free_energy < -599.468


def test_bad_orca_output():

    calc = Calculation(
        name="no_output",
        molecule=test_mol,
        method=method,
        keywords=opt_keywords,
    )

    with pytest.raises(ex.CouldNotGetProperty):
        _ = calc.get_energy()

    with pytest.raises(ex.CouldNotGetProperty):
        _ = calc.get_final_atoms()

    calc.output_file_lines = None
    assert calc.terminated_normally is False


def test_solvation():

    methane = Molecule(
        name="solvated_methane", smiles="C", solvent_name="water"
    )

    with pytest.raises(ex.UnsupportedCalculationInput):

        # Should raise on unsupported calculation type
        method.implicit_solvation_type = "xxx"
        calc = Calculation(
            name="broken_solvation",
            molecule=methane,
            method=method,
            keywords=sp_keywords,
        )
        calc.run()

    method.implicit_solvation_type = "CPCM"
    calc = Calculation(
        name="methane_cpcm",
        molecule=methane,
        method=method,
        keywords=sp_keywords,
    )
    calc.generate_input()

    assert any(
        "cpcm" in line.lower() for line in open("methane_cpcm_orca.inp", "r")
    )
    os.remove("methane_cpcm_orca.inp")

    method.implicit_solvation_type = "SMD"
    calc = Calculation(
        name="methane_smd",
        molecule=methane,
        method=method,
        keywords=sp_keywords,
    )
    calc.generate_input()

    assert any(
        "smd" in line.lower() for line in open("methane_smd_orca.inp", "r")
    )
    os.remove("methane_smd_orca.inp")


def test_vdw_solvent_not_present():
    mol = Molecule(name="mol", smiles="C", solvent_name="2-butanol")

    orca = ORCA()
    orca.implicit_solvation_type = cpcm

    calc = Calculation(
        name="tmp", molecule=mol, method=orca, keywords=sp_keywords
    )

    # Cannot use gaussian charges for 2-butanol
    with pytest.raises(ex.CalculationException):
        calc.generate_input()


@testutils.work_in_zipped_dir(os.path.join(here, "data", "orca.zip"))
def test_gradients():

    h2 = Molecule(name="h2", atoms=[Atom("H"), Atom("H", x=1.0)])
    calc = Calculation(
        name="h2_grad",
        molecule=h2,
        method=method,
        keywords=method.keywords.grad,
    )
    calc.run()
    h2.energy = calc.get_energy()

    delta_r = 1e-8

    # Energy of a finite difference approximation
    h2_disp = Molecule(
        name="h2_disp", atoms=[Atom("H"), Atom("H", x=1.0 + delta_r)]
    )
    calc = Calculation(
        name="h2_disp",
        molecule=h2_disp,
        method=method,
        keywords=method.keywords.grad,
    )
    calc.run()
    h2_disp.energy = calc.get_energy()

    delta_energy = h2_disp.energy - h2.energy  # Ha
    grad = delta_energy / delta_r  # Ha A^-1

    calc = Calculation(
        name="h2_grad",
        molecule=h2,
        method=method,
        keywords=method.keywords.grad,
    )

    calc.run()

    diff = calc.get_gradients()[1, 0] - grad  # Ha A^-1

    # Difference between the absolute and finite difference approximation
    assert np.abs(diff) < 1e-3


@testutils.work_in_zipped_dir(os.path.join(here, "data", "orca.zip"))
def test_mp2_numerical_gradients():

    calc = Calculation(
        name="tmp",
        molecule=Molecule(atoms=xyz_file_to_atoms("tmp_orca.xyz"), charge=-1),
        method=method,
        keywords=method.keywords.grad,
    )
    calc.set_output_filename(filename="tmp_orca.out")

    gradients = calc.get_gradients()
    assert len(gradients) == 6
    expected = (
        np.array([-0.00971201, -0.00773534, -0.02473580]) / Constants.a0_to_ang
    )
    assert np.linalg.norm(expected - gradients[0]) < 1e-6

    # Test for different printing with numerical..
    calc.set_output_filename(filename="numerical_orca.out")
    assert calc.output.filename == "numerical_orca.out"

    gradients = calc.get_gradients()
    assert len(gradients) == 6
    expected = (
        np.array([0.012397372, 0.071726232, -0.070942743])
        / Constants.a0_to_ang
    )
    assert np.linalg.norm(expected - gradients[0]) < 1e-6


@utils.work_in_tmp_dir(filenames_to_copy=[], kept_file_exts=[])
def test_keyword_setting():

    orca = ORCA()
    kwds = orca.keywords.sp
    kwds.functional = "B3LYP"

    # Setter should generate a Functional from the keyword string
    assert isinstance(kwds.functional, Functional)

    calc = Calculation(
        name="tmp", molecule=test_mol.copy(), method=orca, keywords=kwds
    )
    calc.generate_input()
    assert calc.input.exists

    # B3LYP should now be in the in input
    inp_lines = open(calc.input.filename, "r").readlines()
    assert any("B3LYP" in line for line in inp_lines)

    # With a keyword without ORCA defined then raise an exception
    with pytest.raises(ex.UnsupportedCalculationInput):
        orca.keywords.sp.functional = Functional(name="B3LYP", g09="B3LYP")
        calc = Calculation(
            name="tmp",
            molecule=test_mol.copy(),
            method=orca,
            keywords=orca.keywords.sp,
        )
        calc.generate_input()

    # Without a default wavefunction method defined in the single point method
    # we can't set keywords.wf
    with pytest.raises(ValueError):
        orca.keywords.sp.wf_method = "HF"

    # but if we have a WF method in the keywords we should be able to set it
    orca.keywords.sp = SinglePointKeywords(
        [WFMethod("MP2"), BasisSet("def2-TZVP")]
    )

    orca.keywords.sp.wf_method = "HF"
    assert orca.keywords.sp.wf_method == "HF"


@testutils.work_in_zipped_dir(os.path.join(here, "data", "orca.zip"))
def test_hessian_extraction():

    calc = Calculation(
        name="tmp",
        molecule=Molecule(smiles="O"),
        method=method,
        keywords=method.keywords.hess,
    )

    calc.output.filename = "H2O_hess_orca.out"

    with open("H2O_hess_orca.xyz", "w") as xyz_file:
        print(
            "3\n",
            "O     -0.001100    0.363100   -0.000000",
            "H     -0.825000   -0.181900   -0.000000",
            "H      0.826100   -0.181200    0.000000",
            sep="\n",
            file=xyz_file,
        )

    hessian = calc.get_hessian()
    assert hessian.shape == (9, 9)
    # should not have any very large values
    assert np.sum(np.abs(hessian)) < 100

    with pytest.raises(ex.CouldNotGetProperty):
        calc.set_output_filename(filename="no_file.out")

    with pytest.raises(ex.CouldNotGetProperty):
        calc.set_output_filename(filename="H2O_hess_broken.out")


@testutils.work_in_zipped_dir(os.path.join(here, "data", "orca.zip"))
def test_charges_from_v5_output_file():

    water = Molecule(smiles="O")
    calc = Calculation(
        name="h2_grad",
        molecule=water,
        method=method,
        keywords=method.keywords.sp,
    )
    calc.output.filename = "h2o_orca_v5_charges.out"

    assert calc.output.exists

    #                                       q_O        q_H       q_H
    assert calc.get_atomic_charges() == [-0.313189, 0.156594, 0.156594]


def test_unsupported_freq_scaling():

    kwds = HessianKeywords(
        ["Freq", "PBE0", "def2-SVP", "%freq\nscalfreq 0.95\nend"]
    )

    calc = Calculation(
        name="opt", molecule=test_mol, method=method, keywords=kwds
    )

    with pytest.raises(ex.UnsupportedCalculationInput):
        calc.generate_input()


@testutils.work_in_zipped_dir(os.path.join(here, "data", "orca.zip"))
def test_orca_optimiser_from_output_file():

    optimiser = ORCAOptimiser(output_lines=[])
    assert not optimiser.converged
    assert not np.isfinite(optimiser.last_energy_change)

    optimiser = ORCAOptimiser(
        output_lines=open("opt_orca.out", "r").readlines()
    )
    assert optimiser.converged
    assert np.isclose(
        optimiser.last_energy_change.to("Ha"),
        -499.734431042133 - -499.734431061148,
    )


@utils.work_in_tmp_dir()
def test_cartesian_constraints_are_printed():

    idxs = [0, 1]
    with open("tmp.inp", "w") as inp_file:
        mol = Molecule(smiles="O")
        mol.constraints.cartesian = idxs
        print_cartesian_constraints(inp_file, mol)

    lines = "".join(open("tmp.inp", "r").readlines())
    for i in idxs:
        assert "{ C " + str(i) + " C }" in lines


@utils.work_in_tmp_dir()
def test_point_charges_are_printed():

    calc_input = CalculationInput(
        keywords=SinglePointKeywords(), point_charges=[PointCharge(0.1)]
    )
    calc_input.filename = "tmp.inp"

    with open(calc_input.filename, "w") as inp_file:
        print_point_charges(inp_file=inp_file, calc_input=calc_input)

    # should print a separate point charge file
    assert os.path.exists("tmp.pc")


@utils.work_in_tmp_dir()
def test_getting_version_from_blank_output():

    calc = Calculation(
        name="tmp",
        molecule=Molecule(smiles="O"),
        method=method,
        keywords=method.keywords.sp,
    )

    filename = "tmp_orca.out"
    with open(filename, "w") as out_file:
        print("some\ninvalid\noutput", file=out_file)

    calc._executor.output.filename = filename
    assert calc.method.version_in(calc) is not None

    # also ensure that this doesn't correspond to a normal termination
    assert not calc.terminated_normally
    assert not calc.method.terminated_normally_in(calc)
