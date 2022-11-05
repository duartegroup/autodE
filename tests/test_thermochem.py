import os
import pytest
import numpy as np
from autode import Molecule, Atom, Calculation, HessianKeywords
from autode.transition_states import TSguess, TransitionState
from autode.thermochemistry import calculate_thermo_cont
from autode.input_output import xyz_file_to_atoms
from autode.values import Energy
from autode.species import Species
from autode.methods import ORCA, G09
from . import testutils
from autode.thermochemistry.igm import _q_rot_igm, _s_rot_rr, _entropy, _zpe

here = os.path.dirname(os.path.abspath(__file__))

orca = ORCA()
g09 = G09()


@testutils.work_in_zipped_dir(os.path.join(here, "data", "symm.zip"))
def test_symmetry_number():

    assert Molecule().symmetry_number == 1

    assert Molecule("BH3.xyz").symmetry_number == 6
    assert Molecule("C6H6.xyz").symmetry_number == 12
    assert Molecule("CO.xyz").symmetry_number == 1
    assert Molecule("CO2.xyz").symmetry_number == 2
    assert Molecule("H2O.xyz").symmetry_number == 2
    assert Molecule("H3N.xyz").symmetry_number == 3
    assert Molecule(smiles="C").symmetry_number == 12

    # Symmetry numbers aren't calculated for large molecules
    h_100 = Species("tmp", atoms=100 * [Atom("H")], charge=1, mult=1)
    assert h_100.symmetry_number == 1


@testutils.work_in_zipped_dir(os.path.join(here, "data", "thermochem.zip"))
def test_h2o():

    h2o = Molecule(smiles="O")

    calc = Calculation(
        name="tmp", molecule=h2o, method=orca, keywords=orca.keywords.hess
    )
    calc.set_output_filename("H2O_hess_orca.out")
    assert calc.output.exists

    # Check that we cannot calculate the thermochemistry with an undefined
    # method/standard state
    with pytest.raises(ValueError):
        h2o.calc_thermo(calc=calc, lfm_method="an unknown method", sn=1)

    with pytest.raises(ValueError):
        h2o.calc_thermo(calc=calc, ss="1nm", sn=1)

    # Calculate using the default method from ORCA
    h2o.calc_thermo(calc=calc, ss="1atm", sn=1)

    # Ensure the calculated free energy contribution is close to value obtained
    # directly from ORCA
    assert h2o.g_cont is not None
    assert np.isclose(
        h2o.g_cont,
        Energy(0.00327564, units="ha"),
        atol=Energy(0.1, units="kcal mol-1").to("ha"),
    )

    # and likewise for the enthalpy
    assert np.isclose(
        h2o.h_cont,
        Energy(0.02536189087, units="ha"),
        atol=Energy(0.1, units="kcal mol-1").to("ha"),
    )

    # Check that the standard state correction is applied correctly
    h2o_1m = Molecule(smiles="O")
    h2o_1m.calc_thermo(calc=calc, ss="1M", sn=1)

    # with a difference of ~1.9 kcal mol-1 at room temperature
    g_diff = (h2o_1m.g_cont - h2o.g_cont).to("kcal mol-1")
    assert np.isclose(g_diff - Energy(1.9, units="kcal mol-1"), 0.0, atol=0.2)

    # but the enthalpy is the same
    assert np.isclose(h2o_1m.h_cont, h2o.h_cont, atol=1e-6)

    # Cannot calculate any other standard states
    with pytest.raises(ValueError):
        h2o.calc_thermo(calc=calc, ss="1nm3", sn=1)


def test_single_atom():

    f_entropy_g09 = Energy(0.011799 / 298.15, units="Ha")  # T S from g09

    f_atom = Molecule(atoms=[Atom("F")])
    f_atom.calc_thermo()
    f_entropy = (f_atom.h_cont - f_atom.g_cont) / 298.15

    # Ensure the calculated and 'actual' from Gaussian09 are close
    assert np.isclose(f_entropy_g09, f_entropy, atol=2e-5)

    # Ensure the rotational partition functions and entropy are 1 and 0
    assert np.isclose(_q_rot_igm(f_atom, temp=298, sigma_r=0), 1.0)
    assert np.isclose(_s_rot_rr(f_atom, temp=298, sigma_r=0), 0.0)

    assert np.isclose(_zpe(f_atom), 0.0)

    assert np.isclose(f_atom.zpe, 0.0)


def test_no_atoms():

    mol = Molecule()
    assert mol.g_cont is None and mol.h_cont is None

    # Nothing to be calculated for a molecule with no atoms
    calculate_thermo_cont(mol)
    assert mol.g_cont is None and mol.h_cont is None


def test_no_frequencies():

    mol = Molecule(smiles="O")

    # Cannot calculate the vibrational component without vibrational
    # frequencies
    with pytest.raises(ValueError):
        calculate_thermo_cont(mol)

    assert mol.zpe is None


def test_linear_non_linear_rot():

    h2_tri = Molecule(atoms=[Atom("H"), Atom("H", x=1), Atom("H", x=1, y=1)])
    h2_lin = Molecule(atoms=[Atom("H"), Atom("H", x=1), Atom("H", x=2)])

    assert h2_lin.is_linear()

    # Non linear molecules have slightly more entropy than linear ones
    assert _s_rot_rr(h2_tri, temp=298, sigma_r=1) > _s_rot_rr(
        h2_lin, temp=298, sigma_r=1
    )


@testutils.work_in_zipped_dir(os.path.join(here, "data", "thermochem.zip"))
def test_freq_shift():

    # Needs to have lots of atoms so there are frequencies <100 cm-1
    alkane = Molecule(smiles="CCCCCCC")
    alkane_s = alkane.copy()

    calc = Calculation(
        name="tmp", molecule=alkane, method=g09, keywords=g09.keywords.hess
    )
    calc.set_output_filename("C7H16_hess_g09.log")
    assert calc.output.exists

    alkane.calc_thermo(calc=calc, ss="1atm", sn=1, lfm_method="igm")

    alkane_s.calc_thermo(calc=calc, ss="1atm", sn=1, lfm_method="truhlar")

    # Scaling the frequencies to a defined value using truhlar's method should
    # make the entropic contribution less, thus the free energy should be
    # larger (G = H - TS)
    assert alkane.g_cont < alkane_s.g_cont


@testutils.work_in_zipped_dir(os.path.join(here, "data", "thermochem.zip"))
def test_acetylene():

    mol = Molecule("C2H2.xyz")
    assert np.isclose(mol.weight.to("amu"), 26.01565, atol=0.03)

    calc = Calculation(
        name="tmp", molecule=mol, method=g09, keywords=g09.keywords.hess
    )
    calc.set_output_filename("C2H2_hess_g09.log")

    # Calculate the thermochemical contributions in the same way as G09
    mol.calc_thermo(calc=calc, temp=298.150, ss="1atm", lfm_method="igm", sn=1)

    # Check that the vibrational frequencies are similar
    g09_vib_freqs = [
        694.3255,
        694.3255,
        780.9635,
        780.9635,
        2085.2098,
        3430.9110,
        3534.0987,
    ]

    for freq, g09_freq in zip(mol.vib_frequencies, g09_vib_freqs):
        assert np.isclose(freq.to("cm-1"), g09_freq, atol=1.5)

    # Ensure the calculated values are close to the Gaussian 09 values
    assert np.isclose(mol.g_cont.to("Ha"), 0.007734, atol=1e-5)
    assert np.isclose(mol.h_cont.to("Ha"), 0.031043, atol=1e-5)


@testutils.work_in_zipped_dir(os.path.join(here, "data", "thermochem.zip"))
def test_sn2_ts():

    ts = TransitionState(
        TSguess(atoms=xyz_file_to_atoms("TS_sn2.xyz"), charge=-1)
    )

    calc = Calculation(
        name="tmp", molecule=ts, method=g09, keywords=g09.keywords.hess
    )
    calc.set_output_filename("TS_sn2.log")

    ts.calc_thermo(calc=calc, temp=298.15, ss="1atm", sn=1, lfm_method="igm")

    # One 'vibrational' mode is the imaginary frequency which is discarded
    # when calculating thermochemistry
    assert len(ts.vib_frequencies) == 3 * ts.n_atoms - 6 - 1

    # NOTE: Tolerance is 0.3 kcal mol-1 as, for some reason the Gaussian09
    # rotational entropy is not exactly in agreement
    assert np.isclose(ts.g_cont.to("Ha"), 0.010382, atol=5e-4)
    assert np.isclose(ts.h_cont.to("Ha"), 0.042567, atol=5e-4)


@testutils.work_in_zipped_dir(os.path.join(here, "data", "thermochem.zip"))
def test_long_alkane():

    mol = Molecule("alkane_hess_orca.xyz")
    calc = Calculation(
        name="tmp", molecule=mol, method=orca, keywords=orca.keywords.hess
    )

    # Should be able to extract from just a .hess file
    calc.set_output_filename("alkane_hess_orca.hess")

    mol.calc_thermo(
        calc=calc, temp=298.150, ss="1atm", lfm_method="grimme", sn=1
    )

    # Should be close to a previously computed value
    assert np.isclose(mol.g_cont.to("Ha"), 0.2113890180337356, atol=5.2e-4)

    # and <0.5 kcal mol-1 to the ORCA-calculated value
    assert np.isclose(mol.g_cont.to("Ha"), 0.21141149, atol=5e-4)


def test_unknown_entropy_method():

    h2 = Molecule(atoms=[Atom("H"), Atom("H", x=0.7)])

    with pytest.raises(NotImplementedError):
        _ = _entropy(
            species=h2,
            method="an_unkown_method",
            temp=298,
            ss="1M",
            shift=100,
            w0=100,
            alpha=4,
            sigma_r=1,
        )


@testutils.work_in_zipped_dir(os.path.join(here, "data", "thermochem.zip"))
def test_calc_thermo_with_keywords():

    water = Molecule(smiles="O", name="water_pbe")
    water.calc_thermo(
        keywords=HessianKeywords(["PBE", "def2-SVP", "Freq"]), method=orca
    )

    assert os.path.exists("water_pbe_hess_orca.inp")
    inp_line = open("water_pbe_hess_orca.inp", "r").readline()
    assert "PBE " in inp_line

    assert water.enthalpy is not None

    # Ensure the ZPE is close to the expected value
    assert np.isclose(water.zpe.to("Ha"), 0.01952143, atol=1e-5)


@testutils.work_in_zipped_dir(os.path.join(here, "data", "thermochem.zip"))
def test_calc_thermo_with_calc():

    mol = Molecule(smiles="[H][H]", name="h2_calc")

    calc = Calculation(
        "h2_calc_hess",
        method=orca,
        keywords=HessianKeywords(["B3LYP", "def2-SVP", "Freq"]),
        molecule=mol,
    )

    # Should run and non-run calculation
    mol.calc_thermo(calc=calc)
    assert mol.enthalpy is not None

    assert os.path.exists("h2_calc_hess_orca.inp")
    assert "B3LYP " in open("h2_calc_hess_orca.inp", "r").readline()
