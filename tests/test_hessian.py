import os
import pytest
import pickle
import numpy as np
import autode as ade
from autode.utils import work_in_tmp_dir
from . import testutils
import multiprocessing as mp
from autode.config import Config
from autode.atoms import Atom, Atoms
from autode.methods import ORCA, XTB
from autode.calculations import Calculation
from autode.species import Molecule
from autode.values import Frequency
from autode.geom import calc_rmsd
from autode.units import wavenumber, ha_per_ang_sq
from autode.exceptions import CalculationException
from autode.wrappers.keywords import pbe0
from autode.transition_states.base import displaced_species_along_mode
from autode.values import Distance
from autode.wrappers.keywords import HessianKeywords, GradientKeywords
from autode.hessians import (
    Hessian,
    NumericalHessianCalculator,
    HybridHessianCalculator,
)

here = os.path.dirname(os.path.abspath(__file__))
Config.freq_scale_factor = 1.0

# Ha/Å-2
h2o_hessian_arr = np.array(
    [
        [
            2.31423829e00,
            1.56166837e-02,
            8.61890193e-09,
            -1.16433138e00,
            -7.61763557e-01,
            -1.09191486e-09,
            -1.14970123e00,
            7.46143320e-01,
            -7.39260002e-09,
        ],
        [
            1.56179128e-02,
            1.27705582e00,
            -6.14958440e-09,
            -5.68563567e-01,
            -6.43164982e-01,
            -1.65317527e-10,
            5.52895134e-01,
            -6.33819376e-01,
            6.08197843e-09,
        ],
        [
            8.56473769e-09,
            -6.20522983e-09,
            2.38461667e-02,
            -2.33540672e-09,
            1.74823515e-10,
            -9.94345634e-03,
            -6.14782246e-09,
            6.09452763e-09,
            -1.39291346e-02,
        ],
        [
            -1.16340880e00,
            -5.68004778e-01,
            -2.18254923e-09,
            1.21307403e00,
            6.64576848e-01,
            1.49260050e-09,
            -4.97333957e-02,
            -9.65238352e-02,
            7.74806492e-10,
        ],
        [
            -7.61157775e-01,
            -6.42928219e-01,
            1.90093517e-10,
            6.64585946e-01,
            6.09644197e-01,
            8.72800888e-10,
            9.65585874e-02,
            3.32486321e-02,
            -1.08771472e-09,
        ],
        [
            -8.90246292e-10,
            -1.06078463e-10,
            -1.02164725e-02,
            1.50223901e-09,
            8.69409685e-10,
            4.09774275e-02,
            -7.02197099e-10,
            -8.40313571e-10,
            -3.07468472e-02,
        ],
        [
            -1.14872616e00,
            5.52306917e-01,
            -6.27166936e-09,
            -4.97336713e-02,
            9.66013927e-02,
            -8.09399241e-10,
            1.19832228e00,
            -6.48950952e-01,
            6.85905861e-09,
        ],
        [
            7.45507209e-01,
            -6.33561850e-01,
            6.12671503e-09,
            -9.64812852e-02,
            3.32487080e-02,
            -7.74615353e-10,
            -6.48960123e-01,
            6.00277064e-01,
            -5.08741780e-09,
        ],
        [
            -7.51451327e-09,
            6.14650239e-09,
            -1.41604277e-02,
            6.62204039e-10,
            -1.02974704e-09,
            -3.07470051e-02,
            6.85803822e-09,
            -5.09659842e-09,
            4.49197416e-02,
        ],
    ]
)

# Ha/a0^2
co2_hessian_arr = np.array(
    [
        [
            1.1314383525e00,
            4.2385767412e-04,
            3.5051771425e-04,
            -1.0501086627e00,
            -3.7813825173e-04,
            -3.4457384398e-04,
            -8.1229733239e-02,
            -3.7456312285e-05,
            -6.5999542510e-05,
        ],
        [
            4.2325160632e-04,
            3.6570663096e-02,
            1.2516781525e-07,
            -3.8221942577e-04,
            -7.3247574779e-02,
            -1.2460642412e-07,
            -4.1000250417e-05,
            3.6660190261e-02,
            -2.4295086828e-08,
        ],
        [
            3.4996749671e-04,
            1.2517409883e-07,
            3.6556726341e-02,
            -2.8455375094e-04,
            -9.9360220558e-08,
            -7.3235662640e-02,
            -6.5386128105e-05,
            -2.2911454518e-08,
            3.6651071695e-02,
        ],
        [
            -1.0457642155e00,
            -3.8238813778e-04,
            -2.8469599658e-04,
            2.0913922620e00,
            7.8828445803e-04,
            6.8225384914e-04,
            -1.0458280713e00,
            -4.2242186106e-04,
            -2.7748424189e-04,
        ],
        [
            -3.7766992874e-04,
            -7.3411408385e-02,
            -1.0641198331e-07,
            7.8820556787e-04,
            1.4684935286e-01,
            2.6867393467e-07,
            -4.1059908388e-04,
            -7.3404560780e-02,
            -1.1443188980e-07,
        ],
        [
            -3.4413274803e-04,
            -1.3166345414e-07,
            -7.3399496875e-02,
            6.8218586806e-04,
            2.6867411699e-07,
            1.4684780089e-01,
            -3.3810770450e-04,
            -1.4275024213e-07,
            -7.3392638046e-02,
        ],
        [
            -8.1229732348e-02,
            -4.1002276652e-05,
            -6.5388156506e-05,
            -1.0501723903e00,
            -4.1114029467e-04,
            -3.3853608979e-04,
            1.1315021908e00,
            4.6040543323e-04,
            3.4390672878e-04,
        ],
        [
            -3.7453739804e-05,
            3.6660190178e-02,
            -2.2915437653e-08,
            -4.2226004550e-04,
            -7.3240736078e-02,
            -1.3510830784e-07,
            4.5974573122e-04,
            3.6563884025e-02,
            1.3393048024e-07,
        ],
        [
            -6.5997594969e-05,
            -2.4306709511e-08,
            3.6651071540e-02,
            -2.7734157097e-04,
            -1.0679186055e-07,
            -7.3228811671e-02,
            3.4336655361e-04,
            1.3393871147e-02,
            3.6549939293e-02,
        ],
    ]
)


def assert_correct_co2_frequencies(hessian, expected=(666, 1415, 2517)):
    """Ensure the projected frequencies of CO2 are roughly right"""
    nu_1, nu_2, nu_3 = expected

    print(hessian.frequencies_proj)
    assert sum(freq == 0.0 for freq in hessian.frequencies_proj) == 5

    # Should have a degenerate bending mode for CO2 with ν = 666 cm-1
    assert (
        sum(
            np.isclose(Frequency(nu_1, units="cm-1"), freq, atol=2.0)
            for freq in hessian.frequencies_proj
        )
        == 2
    )

    # and two others that are larger
    assert (
        sum(
            np.isclose(Frequency(nu_2, units="cm-1"), freq, atol=2.0)
            for freq in hessian.frequencies_proj
        )
        == 1
    )

    assert (
        sum(
            np.isclose(Frequency(nu_3, units="cm-1"), freq, atol=2.0)
            for freq in hessian.frequencies_proj
        )
        == 1
    )


def test_hessian_class():

    hessian = Hessian(h2o_hessian_arr, units="Ha Å^-2")
    assert "hessian" in repr(hessian).lower()
    assert hash(hessian) is not None

    # Cannot project without atoms
    with pytest.raises(ValueError):
        _ = hessian.frequencies_proj

    with pytest.raises(ValueError):
        _ = hessian.normal_modes_proj

    with pytest.raises(ValueError):
        _ = hessian._proj_matrix

    with pytest.raises(ValueError):
        _ = hessian._mass_weighted

    # without atoms the number of translations/rotations/vibrations is unknown
    with pytest.raises(ValueError):
        assert hessian.n_tr == 6

    with pytest.raises(ValueError):
        assert hessian.n_v == 3

    # Must have matching Hessian and atom dimensions i.e. 3Nx3N for N atoms
    with pytest.raises(ValueError):
        _ = Hessian(h2o_hessian_arr, atoms=[])

    # Check the number of translations and rotations and the number of
    # expected vibrations
    hessian.atoms = Atoms(
        [
            Atom("O", -0.0011, 0.3631, -0.0),
            Atom("H", -0.8250, -0.1819, -0.0),
            Atom("H", 0.8261, -0.1812, 0.0),
        ]
    )

    assert hessian.n_tr == 6
    assert hessian.n_v == 3


def test_hessian_set():

    h2o = Molecule(smiles="O")

    # Cannot set the Hessian as a matrix that isn't 3Nx3N
    with pytest.raises(ValueError):
        h2o.hessian = np.array([])

    with pytest.raises(ValueError):
        h2o.hessian = np.arange(0, 3 * h2o.n_atoms)

    # Hessian must be an array..
    with pytest.raises(ValueError):
        h2o.hessian = 5

    assert h2o.hessian is None
    h2o.hessian = np.zeros(shape=(3 * h2o.n_atoms, 3 * h2o.n_atoms))
    assert h2o.hessian is not None


def test_hessian_freqs():

    h2o = Molecule(smiles="O")
    h2o.hessian = h2o_hessian_arr

    assert isinstance(h2o.hessian, Hessian)
    freqs = h2o.hessian.frequencies

    # Should have 2 frequencies in the 3500 cm-1 range for the O-H stretches
    assert (
        sum(
            [
                Frequency(3000, units=wavenumber)
                < freq
                < Frequency(4000, units=wavenumber)
                for freq in freqs
            ]
        )
        == 2
    )

    # without projection there is an imaginary frequency
    assert sum(freq.is_imaginary for freq in freqs) == 1

    assert h2o.hessian.normal_modes[0].shape == (h2o.n_atoms, 3)

    # Projecting should give frequencies close to those obtained from ORCA
    # the vibrational frequencies are the largest three (non-zero)
    nu_1, nu_2, nu_3 = h2o.hessian.frequencies_proj[-3:]
    assert np.isclose(nu_1, 1567.610851, atol=1.0)
    assert np.isclose(nu_2, 3467.698182, atol=1.0)
    assert np.isclose(nu_3, 3651.462209, atol=1.0)


def test_hessian_scaled_freqs():

    h2o = Molecule(smiles="O")
    h2o.hessian = h2o_hessian_arr

    nu_no_scaling = h2o.hessian.frequencies_proj[-1]

    Config.freq_scale_factor = 0.9
    h2o.hessian = h2o_hessian_arr

    assert np.isclose(
        0.9 * nu_no_scaling, h2o.hessian.frequencies_proj[-1], atol=0.1
    )

    Config.freq_scale_factor = None


def test_hessian_scale_factor():

    Config.freq_scale_factor = None  # Unset..

    h2o = Molecule(smiles="O")
    hessian = Hessian(h2o_hessian_arr, atoms=h2o.atoms, functional=pbe0)

    # 0.96 is the appropriate scale factor for PBE0, also known as PBE1PBE in
    # Gaussian
    assert np.isclose(hessian._freq_scale_factor, 0.96)

    Config.freq_scale_factor = 0.9
    assert np.isclose(hessian._freq_scale_factor, 0.9)

    Config.freq_scale_factor = 1.0
    hessian.functional = None
    assert np.isclose(hessian._freq_scale_factor, 1.0)

    Config.freq_scale_factor = None
    hessian.functional = None
    assert np.isclose(hessian._freq_scale_factor, 1.0)

    Config.freq_scale_factor = 1.0


@testutils.work_in_zipped_dir(os.path.join(here, "data", "hessians.zip"))
def test_hessian_modes():
    """Ensure the translational, rotational and vibrational modes are close
    to the expected values for a projected Hessian"""

    h2o = Molecule("H2O_hess_orca.xyz")
    h2o.hessian = h2o_hessian_arr
    assert h2o.hessian.units == ha_per_ang_sq

    # The structure is a minimum, thus there should be no imaginary frequencies
    assert h2o.imaginary_frequencies is None

    for trans_mode in h2o.hessian.normal_modes_proj[:3]:
        assert np.allclose(trans_mode, np.zeros(shape=(h2o.n_atoms, 3)))

    for rot_mode in h2o.hessian.normal_modes_proj[3:6]:
        assert np.allclose(rot_mode, np.zeros(shape=(h2o.n_atoms, 3)))

    for i, vib_mode in enumerate(h2o.hessian.normal_modes_proj[6:]):

        # Vibrational modes should have no component in the z-axis
        for j, _ in enumerate(h2o.atoms):
            assert np.isclose(vib_mode[j, 2], 0.0, atol=1e-4)

        # and be close to their un-projected analogues for a minimum either
        # forwards or backwards (projection doesn't conserve the direction)
        assert np.allclose(
            vib_mode, h2o.hessian.normal_modes[6 + i], atol=0.1
        ) or np.allclose(vib_mode, -h2o.hessian.normal_modes[6 + i], atol=0.1)

    # Hessian units should be retained
    assert h2o.hessian.units == ha_per_ang_sq


@testutils.work_in_zipped_dir(os.path.join(here, "data", "hessians.zip"))
def test_proj_modes():
    """
    Test the projected normal modes are close to those obtained from an
    ORCA projection. Displaced geometries generated with Chemcraft using a
    scale factor of 0.5
    """
    bend_orca = np.array(
        [
            [-0.001006000, 0.326448000, 0.0],
            [-1.025049000, 0.108976000, 0.0],
            [1.024653500, 0.109669000, 0.0],
        ]
    )

    symm_orca = np.array(
        [
            [0.001894000, 0.340046000, 0.00],
            [-0.547993500, -0.014124500, 0.00],
            [0.501572000, 0.016941000, 0.00],
        ]
    )

    asym_orca = np.array(
        [
            [0.035851500, 0.365107500, 0.0],
            [-1.143189500, -0.391339500, 0.0],
            [0.557796500, -0.003620500, 0.0],
        ]
    )

    h2o = Molecule("H2O_hess_orca.xyz")
    h2o.hessian = h2o_hessian_arr

    for mode_n, coords in zip((6, 7, 8), (bend_orca, symm_orca, asym_orca)):

        bend_f = displaced_species_along_mode(
            h2o, mode_number=mode_n, disp_factor=0.5
        )

        bend_b = displaced_species_along_mode(
            h2o, mode_number=mode_n, disp_factor=-0.5
        )

        # Correct displacement could be either forwards or backwards
        assert (
            calc_rmsd(coords, bend_f.coordinates) < 0.03
            or calc_rmsd(coords, bend_b.coordinates) < 0.03
        )


@testutils.work_in_zipped_dir(os.path.join(here, "data", "hessians.zip"))
def test_hessian_linear_freqs():

    co2 = Molecule("CO2_opt.xyz")
    assert co2.is_linear()

    co2.hessian = Hessian(co2_hessian_arr, units="Ha/a0^2")
    assert_correct_co2_frequencies(hessian=co2.hessian)


@testutils.work_in_zipped_dir(os.path.join(here, "data", "hessians.zip"))
def test_gaussian_hessian_extract_h2():

    h2 = ade.Molecule(
        atoms=[ade.Atom("H", x=0.3804), ade.Atom("H", x=-0.3804)]
    )

    calc = Calculation(
        name="tmp",
        molecule=h2,
        method=ade.methods.G09(),
        keywords=ade.HessianKeywords(),
    )

    calc.set_output_filename("H2_hess_g09.log")
    assert np.isclose(
        h2.hessian.frequencies[-1], Frequency(4383.9811), atol=1.0
    )

    assert np.isclose(
        h2.hessian.frequencies_proj[-1], Frequency(4383.9811), atol=1.0
    )


@testutils.work_in_zipped_dir(os.path.join(here, "data", "hessians.zip"))
def test_gaussian_hessian_extract_co2():

    co2 = Molecule("CO2_opt.xyz")

    calc = Calculation(
        name="tmp",
        molecule=co2,
        method=ade.methods.G09(),
        keywords=ade.HessianKeywords([]),
    )

    calc.set_output_filename("CO2_opt_hess_g09.log")

    assert all(
        np.isclose(freq, Frequency(0, units="cm-1"), atol=10)
        for freq in co2.hessian.frequencies[:5]
    )

    assert all(freq == 0.0 for freq in co2.hessian.frequencies_proj[:5])

    assert_correct_co2_frequencies(hessian=co2.hessian)


@testutils.work_in_zipped_dir(os.path.join(here, "data", "hessians.zip"))
def test_nwchem_hessian_extract_h2o():

    water = ade.Molecule(smiles="O")
    calc = Calculation(
        name="tmp",
        molecule=water,
        method=ade.methods.NWChem(),
        keywords=ade.HessianKeywords(),
    )

    calc.set_output_filename("H2O_hess_nwchem.out")
    hessian = water.hessian

    for freqs in (hessian.frequencies, hessian.frequencies_proj):
        assert sum(np.isclose(freq, 0.0, atol=15) for freq in freqs) == 6

        assert (
            sum(np.isclose(freq, Frequency(1642.78), atol=4) for freq in freqs)
            == 1
        )

        assert (
            sum(np.isclose(freq, Frequency(3860.38), atol=4) for freq in freqs)
            == 1
        )

        assert (
            sum(np.isclose(freq, Frequency(3959.20), atol=4) for freq in freqs)
            == 1
        )


@testutils.work_in_zipped_dir(os.path.join(here, "data", "hessians.zip"))
def test_nwchem_hessian_co2():

    co2 = ade.Molecule(smiles="O=C=O")
    calc = Calculation(
        name="tmp",
        molecule=co2,
        method=ade.methods.NWChem(),
        keywords=ade.HessianKeywords(),
    )
    calc.set_output_filename("CO2_hess_nwchem.out")
    print(co2.hessian)
    print(co2.hessian._mass_weighted)
    assert_correct_co2_frequencies(
        hessian=co2.hessian, expected=(659.76, 1406.83, 2495.73)
    )


@testutils.work_in_zipped_dir(os.path.join(here, "data", "hessians.zip"))
def test_imag_mode():
    """
    Ensure the imaginary mode for an SN2 reaction is close to that obtained
    from ORCA by checking forwards (f) and backwards (b) displaced geometries
    using a factor of 0.5
    """

    ts = Molecule("sn2_TS.xyz", charge=-1)

    calc = Calculation(
        name="tmp", molecule=ts, method=ORCA(), keywords=ORCA().keywords.hess
    )

    calc.set_output_filename("sn2_TS.out")

    assert np.isclose(
        ts.imaginary_frequencies[0], Frequency(-552.64, units="cm-1"), atol=1.0
    )

    imag_f = displaced_species_along_mode(ts, mode_number=6, disp_factor=0.5)
    imag_f_ade = imag_f.coordinates
    imag_b = displaced_species_along_mode(ts, mode_number=6, disp_factor=-0.5)
    imag_b_ade = imag_b.coordinates

    imag_f_orca = np.array(
        [
            [-5.081783000, 4.433438000, 0.062274000],
            [-0.900674000, 4.531897500, -0.036945500],
            [-3.619944500, 4.467834500, 0.028735000],
            [-3.159098000, 3.868992500, -0.899404000],
            [-3.116380000, 3.989669000, 1.004047500],
            [-3.177138500, 5.577882000, -0.051681500],
        ]
    )

    imag_b_orca = np.array(
        [
            [-5.348013000, 4.42739200, 0.06913800],
            [-1.035348000, 4.52875050, -0.03346050],
            [-2.747891500, 4.48803350, 0.00668900],
            [-3.369132000, 3.90162150, -0.83858800],
            [-3.330032000, 4.01514900, 0.94577250],
            [-3.390143500, 5.50372200, -0.04345650],
        ]
    )

    assert (
        calc_rmsd(imag_f_ade, imag_f_orca) < 0.05
        and calc_rmsd(imag_b_ade, imag_b_orca) < 0.05
    ) or (
        calc_rmsd(imag_b_ade, imag_f_orca) < 0.05
        and calc_rmsd(imag_f_ade, imag_b_orca) < 0.05
    )


@testutils.work_in_zipped_dir(os.path.join(here, "data", "hessians.zip"))
def test_extract_wrong_molecule_hessian():

    calc = Calculation(
        name="tmp",
        molecule=ade.Molecule(smiles="[H]"),
        method=ade.methods.G09(),
        keywords=ade.HessianKeywords([]),
    )

    # Should raise an exception if the Hessian extracted is not 3Nx3N for
    # N atoms (1 here)
    with pytest.raises(CalculationException):
        calc.set_output_filename("CO2_opt_hess_g09.log")


def test_num_hess_invalid_input():

    water = Molecule(smiles="O")
    orca = ORCA()

    # Keywords must be GradientKeywords that don't include any 'Hessian'
    # or 'frequency' keywords
    for invalid_kwds in (
        None,
        GradientKeywords(["Freq", "PBE", "Def2-SVP"]),
        HessianKeywords(["PBE", "Def2-SVP"]),
    ):

        with pytest.raises(ValueError):
            nhc = NumericalHessianCalculator(
                species=water,
                method=orca,
                keywords=invalid_kwds,
                do_c_diff=False,
                shift=Distance(1e-3, units="Å"),
            )
            nhc.calculate()


@testutils.work_in_zipped_dir(os.path.join(here, "data", "num_hess.zip"))
def test_h2_hessian():

    h2 = Molecule(name="H2", atoms=[Atom("H"), Atom("H", x=0.77)])

    h2.calc_hessian(method=ORCA(), numerical=False)
    analytic_hessian = h2.hessian.copy()

    h2.hessian = None  # Clear the analytic Hessian

    h2.calc_hessian(method=ORCA(), numerical=True)
    assert h2.hessian is not None

    num_hessian = h2.hessian.copy()

    assert np.allclose(analytic_hessian, num_hessian, atol=1e-2)


@testutils.work_in_zipped_dir(os.path.join(here, "data", "num_hess.zip"))
def test_h2_c_diff_hessian():

    h2 = Molecule(name="H2", atoms=[Atom("H"), Atom("H", x=0.77)])

    h2.calc_hessian(method=ORCA(), numerical=False)
    analytic_hessian = h2.hessian.copy()

    h2.hessian = None  # Clear the analytic Hessian and calculate a numerical
    h2.calc_hessian(
        method=ORCA(), numerical=True, use_central_differences=True
    )

    # Central differences should afford a very good Hessian cf. analytic
    assert np.allclose(analytic_hessian, h2.hessian, atol=1e-3)


@testutils.work_in_zipped_dir(os.path.join(here, "data", "num_hess.zip"))
@testutils.requires_with_working_xtb_install
def test_h2_xtb_vs_orca_hessian():

    h2 = Molecule(name="H2", atoms=[Atom("H"), Atom("H", x=0.77)])

    h2.calc_hessian(method=ORCA(), numerical=False)
    orca_hessian = h2.hessian.copy()

    h2.calc_hessian(method=XTB(), numerical=True, use_central_differences=True)
    xtb_hessian = h2.hessian.copy()

    # ORCA and XTB Hessians should be similar, ish
    assert np.allclose(orca_hessian, xtb_hessian, atol=0.3)


@testutils.work_in_zipped_dir(os.path.join(here, "data", "num_hess.zip"))
@testutils.requires_with_working_xtb_install
def test_ind_num_hess_row():
    """Calculate d^2E/dx0dx0 using numerical displacements with and
    without central differences"""

    h2 = Molecule(name="H2", atoms=[Atom("H"), Atom("H", x=0.77)])
    xtb = XTB()

    for flag in (True, False):

        calculator = NumericalHessianCalculator(
            species=h2,
            method=xtb,
            keywords=xtb.keywords.grad,
            do_c_diff=flag,
            shift=Distance(0.001, units="Å"),
        )

        # Non central differences require an initial gradient at the curr geom
        calculator._init_gradient = calculator._gradient(calculator._species)

        if flag:
            row = calculator._cdiff_row(atom_idx=0, component=0)
        else:
            row = calculator._diff_row(atom_idx=0, component=0)

        assert np.isclose(row[0], 1.00957, atol=1e-1)


def test_partial_num_hess_init():

    # Cannot generate a PartialNumericalHessianCalculator with atom indexes
    # that are not present in the system
    mol = ade.Molecule(smiles="O")

    orca = ORCA()
    orca.path = here  # spoof ORCA install
    assert orca.is_available

    for invalid_idx in (-1, 3, "a"):
        with pytest.raises(ValueError):
            _ = HybridHessianCalculator(
                mol,
                idxs=(invalid_idx,),
                shift=Distance(0.01),
                lmethod=orca,
                hmethod=orca,
            )


@testutils.requires_with_working_xtb_install
@testutils.work_in_zipped_dir(os.path.join(here, "data", "num_hess.zip"))
def test_partial_water_num_hess():

    orca_num_hess = np.array(
        [
            [2.31, 0.01, 0.0, -1.16, -0.76, 0.0, -1.15, 0.75, -0.0],
            [0.01, 1.28, 0.0, -0.57, -0.64, 0.0, 0.55, -0.63, 0.0],
            [0.0, 0.0, 0.03, -0.0, -0.0, -0.01, 0.0, -0.0, -0.02],
            [-1.16, -0.57, -0.0, 1.22, 0.67, -0.0, -0.05, -0.1, 0.0],
            [-0.76, -0.64, -0.0, 0.67, 0.61, -0.0, 0.1, 0.03, 0.0],
            [0.0, 0.0, -0.01, -0.0, -0.0, 0.04, -0.0, 0.0, -0.03],
            [-1.15, 0.55, 0.0, -0.05, 0.1, -0.0, 1.2, -0.65, 0.0],
            [0.75, -0.63, -0.0, -0.1, 0.03, 0.0, -0.65, 0.6, -0.0],
            [-0.0, 0.0, -0.02, 0.0, 0.0, -0.03, 0.0, -0.0, 0.05],
        ]
    )

    xtb_num_hess = np.array(
        [
            [1.85, 0.01, 0.0, -0.93, -0.6, 0.0, -0.92, 0.59, -0.0],
            [0.01, 1.13, 0.0, -0.46, -0.57, 0.0, 0.45, -0.56, 0.0],
            [0.0, 0.0, 0.05, -0.0, -0.0, -0.06, 0.0, -0.0, -0.06],
            [-0.93, -0.46, -0.0, 1.02, 0.53, -0.0, -0.09, -0.07, 0.0],
            [-0.6, -0.57, -0.0, 0.53, 0.51, -0.0, 0.07, 0.06, -0.0],
            [0.0, 0.0, -0.06, -0.0, -0.0, 0.04, -0.0, -0.0, 0.05],
            [-0.92, 0.45, 0.0, -0.09, 0.07, -0.0, 1.0, -0.52, 0.0],
            [0.59, -0.56, -0.0, -0.07, 0.06, -0.0, -0.52, 0.5, -0.0],
            [-0.0, 0.0, -0.06, 0.0, -0.0, 0.05, 0.0, -0.0, 0.05],
        ]
    )

    orca = ORCA()
    orca.path = here  # spoof ORCA install
    assert orca.is_available

    water = Molecule(
        name="water_partial_num_hess",
        charge=0,
        mult=1,
        atoms=[
            Atom("O", -0.00110, 0.36310, -0.00000),
            Atom("H", -0.82500, -0.18190, -0.00000),
            Atom("H", 0.82610, -0.18120, 0.00000),
        ],
    )

    calculator = HybridHessianCalculator(
        water,
        idxs=(0,),
        shift=Distance(0.001, units="Å"),
        hmethod=orca,
        lmethod=XTB(),
    )
    calculator.calculate()
    partial_hess = calculator.hessian
    """
    Partial Hessian should have structure
    
         (   A    B  )
    H =  (           )
         (   B    C  )
    """

    # Block A for the displacement for atom 0 should be identical to the
    # total ORCA hessian
    assert np.allclose(partial_hess[:3, :3], orca_num_hess[:3, :3], atol=1e-2)

    # while block C should be just the XTB numerical Hessian
    assert np.allclose(partial_hess[3:, 3:], xtb_num_hess[3:, 3:], atol=1e-2)

    # who knows what the off diagonals should be...
    assert np.allclose(
        partial_hess[3:, :3],
        (xtb_num_hess[3:, :3] + orca_num_hess[3:, :3]) / 2.0,
        atol=1e-1,
    )


@testutils.requires_with_working_xtb_install
@work_in_tmp_dir()
def test_numerical_hessian_in_daemon():
    """
    Ensure that no exceptions are raised when a numerical hessian is
    calculated within a multiprocessing pool
    """

    with mp.pool.Pool(processes=1) as pool:

        res = pool.apply_async(func=_calc_num_hessian_h2)
        _ = res.get(timeout=None)


def _calc_num_hessian_h2():

    assert mp.current_process().daemon
    h2 = Molecule(smiles="[H][H]")
    h2.calc_hessian(method=XTB(), numerical=True, n_cores=1)


@testutils.requires_with_working_xtb_install
@work_in_tmp_dir()
def test_serial_calculation_matches_parallel():

    h2 = Molecule(atoms=[Atom("H"), Atom("H", x=0.77)])
    xtb = XTB()

    nhc = NumericalHessianCalculator(
        species=h2,
        method=xtb,
        keywords=xtb.keywords.grad,
        do_c_diff=False,
        shift=Distance(0.01, units="Å"),
    )
    nhc.calculate()
    parallel_result = nhc.hessian.copy()
    nhc._calculated_rows.clear()

    nhc._calculate_in_serial()
    serial_result = nhc.hessian

    assert np.allclose(parallel_result, serial_result)


@work_in_tmp_dir()
def test_hessian_pickle_and_unpickle():

    mol = Molecule(smiles="O")
    mol.hessian = Hessian(np.eye(3 * mol.n_atoms), atoms=mol.atoms)

    with open("tmp.obj", "wb") as file:
        pickle.dump(mol.hessian, file=file)

    with open("tmp.obj", "rb") as file:
        reloaded_hessian = pickle.load(file=file)

    assert reloaded_hessian.shape == (3 * mol.n_atoms, 3 * mol.n_atoms)
    assert reloaded_hessian.atoms == mol.atoms
