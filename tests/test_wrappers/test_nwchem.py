from autode import utils
from autode.wrappers.NWChem import NWChem, ecp_block
from autode.point_charges import PointCharge
from autode.calculations import Calculation
from autode.exceptions import UnsupportedCalculationInput, CalculationException
from autode.species.molecule import Molecule
from autode.wrappers.keywords import OptKeywords, SinglePointKeywords
from autode.wrappers.keywords.basis_sets import def2svp
from autode.wrappers.keywords.wf import hf
from autode.wrappers.keywords.functionals import pbe0
from autode.config import Config
from autode.atoms import Atom
from .. import testutils
import numpy as np
import pytest
import os

here = os.path.dirname(os.path.abspath(__file__))
test_mol = Molecule(name="methane", smiles="C")
method = NWChem()
method.path = here  # spoof install

opt_keywords = OptKeywords(
    [
        "basis\n  *   library Def2-SVP\nend",
        "dft\n   xc xpbe96 cpbe96\nend",
        "task dft gradient",
    ]
)


@testutils.work_in_zipped_dir(os.path.join(here, "data", "nwchem.zip"))
def test_opt_calc():

    calc = Calculation(
        name="opt", molecule=test_mol, method=NWChem(), keywords=opt_keywords
    )
    calc.run()

    final_atoms = calc.get_final_atoms()
    assert len(final_atoms) == 5
    assert type(final_atoms[0]) is Atom
    assert -40.4165 < calc.get_energy() < -40.4164
    assert calc.terminated_normally
    assert calc.optimisation_converged()
    assert calc.optimisation_nearly_converged() is False

    # No Hessian is computed for an optimisation calculation
    assert calc.get_hessian() is None

    # Optimisation should result in small gradients
    gradients = calc.get_gradients()
    assert len(gradients) == 5
    assert all(-0.1 < np.linalg.norm(g) < 0.1 for g in gradients)


def test_opt_single_atom():

    h = Molecule(name="H", smiles="[H]")
    calc = Calculation(
        name="opt_h", molecule=h, method=method, keywords=opt_keywords
    )
    calc.generate_input()

    # Can't do an optimisation of a hydrogen atom..
    assert os.path.exists("opt_h_nwchem.nw")
    input_lines = open("opt_h_nwchem.nw", "r").readlines()
    assert "opt" not in [keyword.lower() for keyword in input_lines[0].split()]

    os.remove("opt_h_nwchem.nw")


def test_exception_wf_solvent_calculation():

    solvated_mol = Molecule(name="methane", smiles="C", solvent_name="water")

    calc = Calculation(
        name="opt",
        molecule=solvated_mol,
        method=method,
        keywords=SinglePointKeywords([hf, def2svp]),
    )

    # Cannot have solvent with a non-DFT calculation(?)
    with pytest.raises(CalculationException):
        calc.generate_input()


@testutils.work_in_zipped_dir(os.path.join(here, "data", "nwchem.zip"))
def test_ecp_calc():

    # Should have no ECP block for molecule with only light elements
    water_ecp_block = ecp_block(
        Molecule(smiles="O"), keywords=method.keywords.sp
    )
    assert water_ecp_block == ""

    # Should have no ECP block if the keywords do not define an ECP
    pd_ecp_block = ecp_block(Molecule(smiles="[Pd]"), keywords=OptKeywords([]))
    assert pd_ecp_block == ""

    pdh2 = Molecule(smiles="[H][Pd][H]", name="H2Pd")
    pdh2.single_point(method=method)

    assert os.path.exists("H2Pd_sp_nwchem.nw")
    input_lines = open("H2Pd_sp_nwchem.nw", "r").readlines()
    assert any("ecp" in line for line in input_lines)

    assert pdh2.energy is not None


@testutils.work_in_zipped_dir(os.path.join(here, "data", "nwchem.zip"))
def test_hessian_extract_ts():

    ts = Molecule(
        name="ts",
        atoms=[
            Atom("F", 0.00000, 0.00000, 2.50357),
            Atom("Cl", -0.00000, 0.00000, -1.62454),
            Atom("C", 0.00000, 0.00000, 0.50698),
            Atom("H", 1.05017, 0.24818, 0.60979),
            Atom("H", -0.74001, 0.78538, 0.60979),
            Atom("H", -0.31016, -1.03356, 0.60979),
        ],
        charge=-1,
    )

    calc = Calculation(
        name="sn2_hess",
        molecule=ts,
        keywords=method.keywords.hess,
        method=method,
    )
    calc.set_output_filename("sn2_hess_nwchem.out")

    assert ts.hessian is not None
    assert ts.hessian.shape == (3 * ts.n_atoms, 3 * ts.n_atoms)

    assert ts.gradient is not None
    assert np.isclose(ts.gradient[-1][-1], -0.000588 / 0.529177)


@testutils.work_in_zipped_dir(os.path.join(here, "data", "nwchem.zip"))
def test_hessian_extract_butane():

    Config.freq_scale_factor = 1.0

    butane = Molecule("butane.xyz")
    calc = Calculation(
        name="butane",
        molecule=butane,
        keywords=method.keywords.hess,
        method=method,
    )
    calc.set_output_filename("butane_hess_nwchem.out")

    hess = calc.get_hessian()
    assert hess is not None

    # bottom right corner element should be positive
    assert hess[-1, -1] > 0
    assert np.isclose(hess.frequencies[0].to("cm-1"), -2385.13, atol=3.0)

    assert np.isclose(hess.frequencies[-1].to("cm-1"), 3500.27, atol=3.0)

    calc = Calculation(
        name="butane",
        molecule=Molecule("butane.xyz"),
        keywords=method.keywords.hess,
        method=method,
    )

    with pytest.raises(CalculationException):
        calc.set_output_filename("broken_hessian.out")


@testutils.work_in_zipped_dir(os.path.join(here, "data", "nwchem.zip"))
def test_hf_calculation():

    h2o = Molecule(smiles="O", name="H2O")
    hf_kwds = [def2svp, "task scf"]

    h2o.single_point(method=method, keywords=hf_kwds)

    assert h2o.energy is not None

    #  Solvation is unavalible with HF in v <7.0.2
    h2o_in_water = Molecule(smiles="O", name="H2O_solv", solvent_name="water")

    with pytest.raises(CalculationException):
        h2o_in_water.single_point(method=method, keywords=hf_kwds)

    # Open-shell calculations should be okay

    h = Molecule(smiles="[H]", name="H")
    h.single_point(method=method, keywords=hf_kwds)

    assert np.isclose(h.energy, -0.5, atol=0.001)

    # Should also support other arguments in the SCF block
    hf_kwds = [def2svp, "scf\n    maxiter 100\nend", "task scf"]
    h.single_point(method=method, keywords=hf_kwds)

    assert h.energy is not None


@testutils.work_in_zipped_dir(os.path.join(here, "data", "nwchem.zip"))
def test_point_charge_calculation():

    h = Molecule(smiles="[H]")

    calc = Calculation(
        name="h",
        molecule=h,
        method=method,
        keywords=SinglePointKeywords([def2svp, "task scf"]),
        point_charges=[PointCharge(1.0, 0.0, 0.0, 1.0)],
    )
    calc.run()

    assert calc.get_energy() is not None

    # H atom energy with a point charge should be different from the
    # isolated atoms HF energy
    assert not np.isclose(calc.get_energy(), -0.5, atol=0.001)


@testutils.work_in_zipped_dir(os.path.join(here, "data", "nwchem.zip"))
def test_charge_extract():

    h2o = Molecule(smiles="O")
    calc = Calculation(
        name="tmp", molecule=h2o, keywords=method.keywords.sp, method=method
    )
    calc.set_output_filename("H2O_sp_nwchem.out")

    assert h2o.atomic_symbols == ["O", "H", "H"]
    assert np.allclose(h2o.partial_charges, [-0.801244, 0.397696, 0.403548])


def test_no_driver_in_generated_opt_input():

    opt_str = (
        "driver\n"
        "  gmax 0.0003\n"
        "  grms 0.0001\n"
        "  xmax 0.004\n"
        "  xrms 0.002\n"
        "  eprec 0.000005\n"
        "end"
    )

    calc = Calculation(
        name="tmp",
        molecule=Molecule(smiles="O"),
        keywords=OptKeywords([pbe0, def2svp, opt_str]),
        method=method,
    )

    with pytest.raises(UnsupportedCalculationInput):
        calc.generate_input()
