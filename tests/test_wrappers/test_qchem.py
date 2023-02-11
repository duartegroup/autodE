import os
import pytest
import numpy as np
from autode.point_charges import PointCharge
from autode.wrappers.QChem import QChem
from autode.calculations import Calculation
from autode.atoms import Atom
from autode.config import Config
from autode.wrappers.keywords import cpcm, pbe0, def2svp
from autode.species.molecule import Molecule
from autode.wrappers.keywords import SinglePointKeywords, OptKeywords
from autode.utils import work_in_tmp_dir
from autode.exceptions import CalculationException
from ..testutils import work_in_zipped_dir

here = os.path.dirname(os.path.abspath(__file__))
qchem_data_zip_path = os.path.join(here, "data", "qchem.zip")

method = QChem()


def _blank_calc(name="test"):
    """Blank calculation of a single H atom"""

    calc = Calculation(
        name=name,
        molecule=Molecule(atoms=[Atom("H")], mult=2),
        method=method,
        keywords=SinglePointKeywords(),
    )

    return calc


def _completed_thf_calc():

    calc = _blank_calc()
    calc.output.filename = "smd_thf.out"

    assert calc.output.exists
    assert len(calc.output.file_lines) > 0

    return calc


def _custom_output_calc(*lines):
    calc = _blank_calc()
    with open("tmp.out", "w") as out_file:
        print(*lines, sep="\n", file=out_file)

    calc.output.filename = "tmp.out"
    assert calc.output.exists

    return calc


def _broken_output_calc():
    return _custom_output_calc("a", "broken", "output", "file")


def _broken_output_calc2():
    return _custom_output_calc("broken", "Total energy")


def test_base_method():

    assert "qchem" in repr(method).lower()

    # TODO: Implement, if it's useful
    with pytest.raises(Exception):
        _ = method.partial_charges_from(_blank_calc())

    calc = _blank_calc()
    calc.input.point_charges = [PointCharge(0.1, x=0, y=0, z=1)]

    # TODO: Implement point charges within the surroundings of a molecule
    with pytest.raises(NotImplementedError):
        calc.generate_input()


def test_in_out_name():

    calc = _blank_calc(name="test")._executor
    assert method.input_filename_for(calc) == "test_qchem.in"
    assert method.output_filename_for(calc) == "test_qchem.out"


@work_in_zipped_dir(qchem_data_zip_path)
def test_version_extract():

    # Version extraction from a blank calculation should not raise an error
    assert isinstance(method.version_in(_blank_calc()), str)

    version = method.version_in(calc=_completed_thf_calc())

    assert version == "5.4.1"


@work_in_zipped_dir(qchem_data_zip_path)
def test_version_extract_broken_output_file():

    # Should not raise an exception
    version = method.version_in(_broken_output_calc())
    assert isinstance(version, str)


@work_in_zipped_dir(qchem_data_zip_path)
def test_thf_calc_terminated_normally():

    assert _completed_thf_calc().terminated_normally


@work_in_zipped_dir(qchem_data_zip_path)
def test_terminated_abnormally():

    # Without any output the calculation cannot have terminated normally
    calc = _blank_calc()
    assert not method.terminated_normally_in(calc)
    assert not calc.terminated_normally

    # A broken file containing one fewer H atom for an invalid 2S+1
    calc.output.filename = "smd_thf_broken.out"
    assert calc.output.exists
    assert not calc.terminated_normally

    # If the output is not a QChem output file..
    with open("tmp.out", "w") as tmp_out_file:
        print("not", "a", "out", "file", sep="\n", file=tmp_out_file)

    calc = _blank_calc()
    calc.output.filename = "tmp.out"
    assert calc.output.exists

    assert not calc.terminated_normally

    os.remove("tmp.out")


def test_blank_input_generation():

    calc = _blank_calc()
    calc.input.filename = None

    with pytest.raises(ValueError):
        method.generate_input_for(calc=calc)


@work_in_tmp_dir(filenames_to_copy=[], kept_file_exts=[])
def test_unsupported_keywords():

    calc = Calculation(
        name="test",
        molecule=Molecule(atoms=[Atom("H")], mult=2),
        method=method,
        keywords=SinglePointKeywords(
            "$rem\n" "method b3lyp\n" "basis 6-31G*\n" "$end"
        ),
    )

    # Having $blocks in the keywords is not supported
    with pytest.raises(Exception):
        calc.generate_input()


def test_simple_input_generation():

    expected_inp = (
        "$molecule\n"
        "0 2\n"
        "H    0.00000000   0.00000000   0.00000000 \n"
        "$end\n"
        "$rem\n"
        "method pbe0\n"
        "basis def2-SVP\n"
        "$end\n"
    )

    # Simple PBE0/def2-SVP calculation of a hydrogen atom
    h_atom = Molecule(atoms=[Atom("H")], mult=2)
    calc = Calculation(
        name="H_atom",
        molecule=h_atom,
        method=method,
        keywords=SinglePointKeywords([pbe0, def2svp]),
    )

    # Generate the required input
    calc.input.filename = "test.in"
    method.generate_input_for(calc)

    # Remove any blank lines from the input file for comparison, as they are
    # ignored
    inp_lines = [
        line
        for line in open(calc.input.filename, "r").readlines()
        if line != "\n"
    ]
    assert "".join(inp_lines) == expected_inp

    os.remove("test.in")


@work_in_zipped_dir(qchem_data_zip_path)
def test_energy_extraction():

    calc = _completed_thf_calc()
    energy = calc.get_energy()

    assert np.isclose(energy.to("Ha"), -232.45463628, atol=1e-8)

    for calc in (_blank_calc(), _broken_output_calc(), _broken_output_calc2()):
        with pytest.raises(CalculationException):
            _ = method.energy_from(calc)


def _file_contains_one(filename, string):
    """A file contains one line that is an exact match to a string"""
    _list = [line.strip() for line in open(filename, "r")]
    print(_list)
    return sum(item.lower() == string for item in _list) == 1


def _tmp_input_contains(string):
    flag = _file_contains_one(filename="tmp.in", string=string)
    os.remove("tmp.in")
    return flag


def test_jobtype_inference():
    """Check that the jobtype can be infered from the keyword type"""

    def kwd_type_has_job_type(kwd_type, job_type, remove_explicit=True):

        calc = _blank_calc()
        calc.molecule = Molecule(atoms=[Atom("H"), Atom("H", x=0.77)])

        keywords = getattr(method.keywords, kwd_type)

        if remove_explicit:
            # Remove any explicit declaration of the job type in the keywords
            keywords = keywords.__class__(
                [w for w in keywords if "jobtype" not in w.lower()]
            )

        calc.input.keywords = keywords

        with QChem._InputFileWriter("tmp.in") as inp_file:
            inp_file.add_rem_block(calc)

        return _tmp_input_contains(f"jobtype {job_type}")

    assert kwd_type_has_job_type("opt", "opt")
    assert kwd_type_has_job_type("low_opt", "opt")
    assert kwd_type_has_job_type("opt_ts", "ts", remove_explicit=False)

    assert kwd_type_has_job_type("grad", "force")
    assert kwd_type_has_job_type("hess", "freq")


def test_ecp_writing():

    calc = _blank_calc()
    calc.input.keywords = method.keywords.sp

    def write_tmp_input():
        with QChem._InputFileWriter("tmp.in") as inp_file:
            inp_file.add_rem_block(calc)

    # No ECP for a H atom
    assert calc.molecule.n_atoms == 1 and calc.molecule.atoms[0].label == "H"
    write_tmp_input()
    assert not _tmp_input_contains("ecp def2-ecp")

    # Should add an ECP for lead
    calc.molecule = Molecule(atoms=[Atom("Pb")])
    write_tmp_input()
    assert _tmp_input_contains("ecp def2-ecp")


@work_in_zipped_dir(qchem_data_zip_path)
def test_h2o_opt():
    """Check that the energy and geometry is extracted correctly"""

    h2o = Molecule(smiles="O")
    h2o.optimise(method=method)

    assert h2o.energy is not None
    assert np.isclose(h2o.energy.to("Ha"), -76.2766126261376, atol=1e-8)

    assert np.isclose(h2o.distance(0, 1).to("Å"), 0.962586, atol=1e-5)

    assert np.isclose(h2o.angle(1, 0, 2).to("deg"), 103.154810, atol=1e-3)


@work_in_zipped_dir(qchem_data_zip_path)
def test_gradient_extraction_h2o():
    calc = Calculation(
        name="test",
        molecule=Molecule(smiles="O"),
        method=method,
        keywords=OptKeywords(),
    )

    calc.set_output_filename("H2O_opt_qchem.out")

    assert calc.output.exists

    grad = calc.get_gradients()
    assert grad.shape == (3, 3)

    # The minimum should have a gradient close to zero
    assert np.allclose(grad, np.zeros(shape=(3, 3)), atol=1e-4)

    # also for this calculation the optimisation has converged
    assert calc.optimiser.converged
    assert not calc.optimisation_nearly_converged()


@work_in_zipped_dir(qchem_data_zip_path)
def test_gradient_extraction_h2():

    calc = _blank_calc()
    calc.molecule = Molecule(atoms=[Atom("H"), Atom("H", x=0.77)])
    calc.output.filename = "H2_qchem.out"

    grad = calc.get_gradients()
    assert grad.shape == (2, 3)


@work_in_zipped_dir(qchem_data_zip_path)
def test_butane_gradient_extraction():

    calc = _blank_calc()
    calc.output.filename = "partial_C4H10_opt_qchem.out"
    calc.molecule = Molecule(smiles="CCCC")

    assert calc.molecule.n_atoms == 14

    grad = method.gradient_from(calc)
    assert grad.shape == (14, 3)


@work_in_zipped_dir(qchem_data_zip_path)
def test_h2o_hessian_extraction():

    calc = _blank_calc()
    calc.input.keywords = method.keywords.hess
    calc.output.filename = "H2O_hess_qchem.out"
    calc.molecule = Molecule(smiles="O")

    hess = method.hessian_from(calc)
    assert hess.shape == (9, 9)

    # Check the first element is close to that of an ORCA-derived equiv.
    # in Ha / Å-2
    assert np.isclose(hess[0, 0], 2.31423829e00, atol=0.1)

    # Final atoms are available, as the same ones input
    assert np.allclose(
        calc.molecule.coordinates,
        calc.get_final_atoms().coordinates,
        atol=1e-8,
    )


def test_broken_hessian_extraction():
    calc = _broken_output_calc()

    with pytest.raises(CalculationException):
        _ = method.hessian_from(calc)

    calc = _custom_output_calc(
        "some", "output", "then", "Mass-Weighted Hessian Matrix", "X"
    )

    with pytest.raises(CalculationException):
        _ = method.hessian_from(calc)

    if os.path.exists("tmp.out"):
        os.remove("tmp.out")


def test_broken_gradient_extraction():

    calc = _broken_output_calc()

    with pytest.raises(CalculationException):
        _ = method.gradient_from(calc)

    calc = _custom_output_calc(
        "some", "output", "then", "Mass-Weighted Hessian Matrix", "X"
    )

    with pytest.raises(CalculationException):
        _ = method.gradient_from(calc)

    if os.path.exists("tmp.out"):
        os.remove("tmp.out")


@work_in_zipped_dir(qchem_data_zip_path)
def test_calc_terminated_normally_max_opt_cycles():

    # h2o.optimise(method=ade.methods.QChem(),
    # keywords=['method pbe', 'basis def2-SVP', 'geom_opt_max_cycle 2'])

    calc = _blank_calc()
    calc.output.filename = "H2O_opt_max_2_cycles.out"
    assert calc.output.exists

    # Even with a 'fatal error' in the output the calculation was ok
    assert calc.terminated_normally


@work_in_zipped_dir(qchem_data_zip_path)
def test_ts_opt():

    Config.freq_scale_factor = 1.0

    ts_mol = Molecule(
        name="ts",
        charge=-1,
        mult=1,
        solvent_name="water",
        atoms=[
            Atom("F", -4.17085, 3.55524, 1.59944),
            Atom("Cl", -0.75962, 3.53830, -0.72354),
            Atom("C", -2.51988, 3.54681, 0.47836),
            Atom("H", -3.15836, 3.99230, -0.27495),
            Atom("H", -2.54985, 2.47411, 0.62732),
            Atom("H", -2.10961, 4.17548, 1.25945),
        ],
    )

    for pair in [(0, 2), (1, 2)]:
        ts_mol.graph.add_active_edge(*pair)

    calc = Calculation(
        name="sn2_ts",
        molecule=ts_mol,
        method=method,
        keywords=method.keywords.opt_ts,
        n_cores=4,
    )

    # Should skip calculation for already completed and saved calculation
    calc.run()

    assert np.isclose(calc.get_energy().to("Ha"), -599.4788133790, atol=1e-8)

    ts_mol.hessian = calc.get_hessian()
    assert ts_mol.hessian is not None

    assert sum(freq.is_imaginary for freq in ts_mol.vib_frequencies) == 1

    # Should have a single imaginary frequency, ~511 cm-1
    assert np.isclose(ts_mol.vib_frequencies[0].to("cm-1"), -511, atol=2)


@work_in_zipped_dir(qchem_data_zip_path)
def test_constrained_distance_opt():

    mol = Molecule(
        name="water_const_opt",
        atoms=[
            Atom("O", -0.0011, 0.3631, -0.0000),
            Atom("H", -0.8250, -0.1819, -0.0000),
            Atom("H", 0.8261, -0.1812, 0.0000),
        ],
    )

    # Constrain the O-H distance to 0.9 Å
    mol.constraints.distance = {(0, 1): 0.9}
    mol.optimise(method=method)

    assert np.isclose(mol.distance(0, 1).to("Å"), 0.9, atol=1e-2)


@work_in_zipped_dir(qchem_data_zip_path)
def test_constrained_cartesian_opt():
    mol = Molecule(
        name="water_const_opt2",
        atoms=[
            Atom("O", -0.0011, 0.3631, -0.0000),
            Atom("H", -0.8250, -0.1819, -0.0000),
            Atom("H", 0.8261, -0.1812, 0.0000),
        ],
    )

    init_dist0, init_dist1 = mol.distance(0, 1), mol.distance(0, 2)

    mol.constraints.cartesian = [0, 1]
    mol.optimise(method=method)

    # First O-H distance should be unchanged
    assert np.isclose(mol.distance(0, 1), init_dist0, atol=1e-3)

    # while the other O-H distance will relax if atom 2 can move
    assert not np.isclose(mol.distance(0, 2), init_dist1, atol=1e-3)


@work_in_tmp_dir(filenames_to_copy=[], kept_file_exts=[])
def test_opt_single_atom():

    calc = _blank_calc()
    calc.molecule = Molecule(name="H", mult=2, atoms=[Atom("H")])

    calc.input.keywords = method.keywords.opt

    calc.name = "tmp"
    calc.input.filename = "tmp.in"

    method.generate_input_for(calc=calc)
    assert os.path.exists("tmp.in")

    # A single atom cannot be optimised so there should be no opt in the input
    assert not any("opt" in line.lower() for line in open("tmp.in"))


@work_in_tmp_dir(filenames_to_copy=[], kept_file_exts=[])
def test_unsupported_solvent_type():

    calc = _blank_calc()

    # Cannot generate a calculation with an unsupported solvent type
    calc.input.keywords.append(cpcm)

    with pytest.raises(CalculationException):
        calc.generate_input()


@work_in_zipped_dir(qchem_data_zip_path)
def test_butane_grad_extract():

    calc = _blank_calc()
    calc.molecule = Molecule(smiles="CCCC")
    calc.set_output_filename("C4H10_sp_qchem.out")

    assert calc.molecule.energy is not None

    grad = calc.molecule.gradient
    assert grad is not None
    flat_grad = grad.to("Ha a0^-1").flatten()

    # Check the final element of the gradient is as expected
    assert np.isclose(flat_grad[-1], 0.0055454, atol=1e-5)

    assert np.isclose(flat_grad[5], 0.0263383, atol=1e-5)


@work_in_zipped_dir(qchem_data_zip_path)
def test_h2_coordinate_extraction_qchem_v6():

    calc = _blank_calc()
    calc.input.keywords = method.keywords.opt
    calc.molecule = Molecule(smiles="[H][H]")
    calc.set_output_filename("H2_opt_qchem6.out")

    assert np.allclose(
        calc.molecule.coordinates.to("Å"),
        np.array([[+0.3803762086, 0.0, 0.0], [-0.3803762086, 0.0, 0.0]]),
    )


def test_coordinate_extract_from_single_point_calculation():

    calc = _blank_calc()
    calc.input.keywords = method.keywords.sp
    calc.molecule = Molecule(smiles="O")
    init_coords = calc.molecule.coordinates.copy()

    assert np.allclose(init_coords, QChem().coordinates_from(calc))


def test_coordinate_extract_from_single_atom_calculation():

    calc = _blank_calc()
    calc.input.keywords = method.keywords.opt
    calc.molecule = Molecule(atoms=[Atom("H", x=0, y=0, z=0)])

    assert np.allclose(QChem().coordinates_from(calc), np.zeros(3))
