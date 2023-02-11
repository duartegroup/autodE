from typing import List

import numpy as np
import os
import pytest

from autode.utils import work_in_tmp_dir
from autode.atoms import Atom
from autode.wrappers.XTB import XTB
from autode.calculations import Calculation
from autode.species.molecule import Molecule
from autode.point_charges import PointCharge
from autode.exceptions import CalculationException
from autode.wrappers.methods import ExternalMethodEGH
from autode.wrappers.keywords import OptKeywords, SinglePointKeywords
from autode.config import Config
from autode.hessians import Hessian
from autode.values import Coordinates, Gradient, PotentialEnergy
from .. import testutils

here = os.path.dirname(os.path.abspath(__file__))

method = XTB()


@testutils.work_in_zipped_dir(os.path.join(here, "data", "xtb.zip"))
def test_xtb_calculation():

    test_mol = Molecule(
        name="test_mol", smiles="O=C(C=C1)[C@@](C2NC3C=C2)([H])[C@@]3([H])C1=O"
    )
    calc = Calculation(
        name="opt",
        molecule=test_mol,
        method=method,
        keywords=Config.XTB.keywords.opt,
    )
    calc.run()

    assert os.path.exists("opt_xtb.xyz") is True
    assert os.path.exists("opt_xtb.out") is True
    assert test_mol.n_atoms == 22
    assert test_mol.energy == -36.990267613593
    assert calc.output.exists
    assert calc.output.file_lines is not None
    assert calc.input.filename == "opt_xtb.xyz"
    assert calc.output.filename == "opt_xtb.out"
    assert calc.optimiser.converged

    with pytest.raises(NotImplementedError):
        _ = calc.optimiser.last_energy_change

    charges = test_mol.partial_charges
    assert len(charges) == 22
    assert all(-1.0 < c < 1.0 for c in charges)

    test_mol.constraints.update(distance={(0, 1): 1.2539792}, cartesian=[0])

    const_opt = Calculation(
        name="const_opt",
        molecule=test_mol,
        method=method,
        keywords=Config.XTB.keywords.opt,
    )

    const_opt.generate_input()
    assert os.path.exists("const_opt_xtb.xyz")
    assert os.path.exists("xcontrol_const_opt_xtb")

    const_opt.clean_up(force=True)
    assert not os.path.exists("xcontrol_const_opt_xtb")

    # Write an empty output file
    open("tmp.out", "w").close()
    const_opt.output.filename = "tmp.out"

    # cannot get atoms from an empty file
    with pytest.raises(CalculationException):
        const_opt._executor.set_properties()


@testutils.work_in_zipped_dir(os.path.join(here, "data", "xtb.zip"))
def test_energy_extract_no_energy():

    calc = Calculation(
        name="opt",
        molecule=Molecule(smiles="[H][H]"),
        method=method,
        keywords=Config.XTB.keywords.sp,
    )

    # Output where the energy is not present
    calc.output.filename = "h2_sp_xtb_no_energy.out"

    with pytest.raises(CalculationException):
        calc._executor.set_properties()


@testutils.work_in_zipped_dir(os.path.join(here, "data", "xtb.zip"))
def test_point_charge():

    test_mol = Molecule(name="test_mol", smiles="C")

    # Methane with a point charge fairly far away
    calc = Calculation(
        name="opt_point_charge",
        molecule=test_mol,
        method=method,
        keywords=Config.XTB.keywords.opt,
        point_charges=[PointCharge(charge=1.0, x=10, y=1, z=1)],
    )
    calc.run()

    assert -4.178 < test_mol.energy < -4.175


@testutils.work_in_zipped_dir(os.path.join(here, "data", "xtb.zip"))
def test_gradients():

    h2 = Molecule(name="h2", atoms=[Atom("H"), Atom("H", x=1.0)])
    h2.single_point(method)

    delta_r = 1e-5
    h2_disp = Molecule(
        name="h2_disp", atoms=[Atom("H"), Atom("H", x=1.0 + delta_r)]
    )
    h2_disp.single_point(method)

    delta_energy = h2_disp.energy - h2.energy  # Ha
    grad = delta_energy / delta_r  # Ha A^-1

    calc = Calculation(
        name="h2_grad",
        molecule=h2,
        method=method,
        keywords=method.keywords.grad,
    )

    calc.run()

    diff = h2.gradient[1, 0] - grad  # Ha A^-1

    # Difference between the absolute and finite difference approximation
    assert np.abs(diff) < 1e-5

    # Older xtb version
    with open(f"methane_OLD.grad", "w") as gradient_file:
        print(
            "$gradient\n"
            "cycle =      1    SCF energy =    -4.17404780397   |dE/dxyz| =  0.027866\n"
            "3.63797523123375     -1.13138130908142     -0.00032759661848      C \n"
            "5.72449332438353     -1.13197561185651      0.00028950521969      H \n"
            " 2.94133258016711      0.22776472016180     -1.42078243039077      H \n"
            " 2.94175598539510     -0.58111835182372      1.88747566982948      H \n"
            "2.94180792167968     -3.04156357656436     -0.46665514803992      H \n"
            "-1.7221823521705E-05   7.9930724499610E-05  -1.1737079840097E-04\n"
            " 1.4116296505865E-02  -4.0359524399270E-05   3.9719638516747E-05\n"
            "-4.7199424681741E-03   9.0086220034949E-03  -9.4114548523723E-03\n"
            "-4.6956970257351E-03   3.6356853660431E-03   1.2558467871909E-02\n"
            " -4.6834351884340E-03  -1.2683878569638E-02  -3.0693618596526E-03\n"
            "$end",
            file=gradient_file,
        )

    calc = Calculation(
        name="methane",
        molecule=Molecule(name="methane", smiles="C"),
        method=method,
        keywords=method.keywords.grad,
    )
    gradients = method.gradient_from(calc)

    assert gradients.shape == (5, 3)
    assert np.abs(gradients[0, 0]) < 1e-3


@testutils.work_in_zipped_dir(os.path.join(here, "data", "xtb.zip"))
def test_xtb_6_3_2():

    mol = Molecule(name="CH3Cl", smiles="ClC")
    calc = Calculation(
        name="test", molecule=mol, method=method, keywords=method.keywords.opt
    )

    calc.set_output_filename("xtb_6_3_2_opt.out")
    assert mol.n_atoms == 5
    assert np.isclose(mol.atoms[-2].coord[1], -0.47139030225766)


@testutils.work_in_zipped_dir(os.path.join(here, "data", "xtb.zip"))
def test_xtb_6_1_old():

    mol = Molecule(name="methane", smiles="C")
    calc = Calculation(
        name="test", molecule=mol, method=method, keywords=method.keywords.opt
    )

    # TODO: check this extracts the right numbers
    for filename in ("xtb_6_1_opt.out", "xtb_no_version_opt.out"):

        calc.set_output_filename(filename)

        assert set([atom.label for atom in mol.atoms]) == {"C", "H"}
        assert 0.9 < mol.distance(0, 1) < 1.2


class XTBautodEOpt(ExternalMethodEGH, XTB):

    __test__ = False

    def __init__(self):
        ExternalMethodEGH.__init__(
            self,
            executable_name="xtb",
            doi_list=[],
            implicit_solvation_type=None,
            keywords_set=XTB().keywords,
        )

    def _energy_from(self, calc: "CalculationExecutor") -> PotentialEnergy:
        return XTB._energy_from(self, calc)

    def gradient_from(self, calc: "CalculationExecutor") -> Gradient:
        return XTB.gradient_from(self, calc)

    def hessian_from(
        self, calc: "autode.calculations.executors.CalculationExecutor"
    ) -> Hessian:
        pass

    def coordinates_from(self, calc: "CalculationExecutor") -> Coordinates:
        pass

    def partial_charges_from(self, calc: "CalculationExecutor") -> List[float]:
        pass

    def terminated_normally_in(self, calc: "CalculationExecutor") -> bool:
        return True

    def version_in(self, calc: "CalculationExecutor") -> str:
        pass

    @staticmethod
    def input_filename_for(calc: "CalculationExecutor") -> str:
        return XTB.input_filename_for(calc)

    @staticmethod
    def output_filename_for(calc: "CalculationExecutor") -> str:
        return XTB.output_filename_for(calc)

    def generate_input_for(self, calc: "CalculationExecutor") -> None:

        calc.molecule.print_xyz_file(filename=calc.input.filename)
        return None

    def __repr__(self):
        return XTB.__repr__(self)


@testutils.requires_with_working_xtb_install
@work_in_tmp_dir()
def test_xtb_with_autode_opt_method():

    mol = Molecule(smiles="C")
    calc = Calculation(
        name="methane",
        molecule=mol,
        method=XTBautodEOpt(),
        keywords=OptKeywords(),
    )
    calc.run()

    assert calc.optimiser.converged


@testutils.requires_with_working_xtb_install
@work_in_tmp_dir()
def test_xtb_with_autode_opt_method_for_a_single_atom():

    mol = Molecule(atoms=[Atom("H")], mult=2)
    calc = Calculation(
        name="h_atom",
        molecule=mol,
        method=XTBautodEOpt(),
        keywords=OptKeywords(),
    )
    calc.run()

    assert calc.optimiser.converged
    assert mol.energy is not None


@testutils.requires_with_working_xtb_install
@work_in_tmp_dir()
def test_xtb_opt_non_contiguous_range_cart_constraints():

    mol = Molecule(smiles="CC")
    mol.constraints.cartesian = [0, 1, 2, 5]

    calc = Calculation(
        name="ethane", molecule=mol, method=XTB(), keywords=OptKeywords()
    )
    calc.run()

    assert len(calc.input.additional_filenames) > 0
    xcontrol_lines = open(calc.input.additional_filenames[-1], "r").readlines()
    expected_range = "1-3,6"
    assert sum(expected_range in line for line in xcontrol_lines) == 1

    assert calc.optimiser.converged
    assert mol.energy is not None


@testutils.requires_with_working_xtb_install
@work_in_tmp_dir()
def test_xtb_errors_with_infinite_nuclear_repulsion():

    # H2 with a zero H-H distance
    mol = Molecule(atoms=[Atom("H"), Atom("H")])
    calc = Calculation(
        name="h_atom",
        molecule=mol,
        method=XTB(),
        keywords=SinglePointKeywords(),
    )

    with pytest.raises(CalculationException):
        calc.run()


@work_in_tmp_dir()
def test_xtb_did_not_terminate_normally_with_blank_output():

    mol = Molecule(atoms=[Atom("H")], mult=2)
    calc = Calculation(
        name="h_atom",
        molecule=mol,
        method=XTB(),
        keywords=SinglePointKeywords(),
    )

    with open("tmp.out", "w") as file:
        print("\n", file=file)

    calc._executor.output.filename = "tmp.out"
    assert not calc.method.terminated_normally_in(calc)


@testutils.requires_with_working_xtb_install
@work_in_tmp_dir()
def test_ade_opt_rerun_with_different_input_skip_saved_opt():
    def run_calc(_mol):
        calc = Calculation(
            name="water",
            molecule=_mol,
            method=XTBautodEOpt(),
            keywords=OptKeywords(),
        )
        calc.run()

    mol = Molecule(smiles="O")
    run_calc(mol)

    unconstrained_energy = mol.energy.copy()

    mol.constraints.distance = {(0, 1): 0.9}
    run_calc(mol)

    assert mol.energy != unconstrained_energy


@testutils.requires_with_working_xtb_install
@work_in_tmp_dir()
def test_xtb_cartesian_constrained_opt():

    init_r = 0.9
    h2 = Molecule(atoms=[Atom("H"), Atom("H", x=init_r)])

    h2_unconstrained = h2.new_species(name="unconstrained_h2")
    h2_unconstrained.optimise(method=XTB())
    # expected minimum for H2 is ~0.77 Å
    assert abs(h2_unconstrained.distance(0, 1) - init_r) > 0.1

    h2.constraints.cartesian = [0, 1]
    h2.optimise(method=XTB())

    # if the coordinates are constrained then the distance should be
    # close to the initial
    assert abs(h2.distance(0, 1) - init_r) < 0.1
