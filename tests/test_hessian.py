import pytest
import numpy as np
import autode as ade
from autode.calculation import Calculation
from autode.species import Molecule
from autode.values import Frequency
from autode.thermo.hessians import Hessian
from autode.units import wavenumber

# Ha/Å-2
h2o_hessian_arr = np.array([[2.31423829e+00,  1.56166837e-02,  8.61890193e-09,
                           -1.16433138e+00, -7.61763557e-01, -1.09191486e-09,
                           -1.14970123e+00,  7.46143320e-01, -7.39260002e-09],
                         [1.56179128e-02,  1.27705582e+00, -6.14958440e-09,
                           -5.68563567e-01, -6.43164982e-01, -1.65317527e-10,
                           5.52895134e-01, -6.33819376e-01, 6.08197843e-09],
                         [8.56473769e-09, -6.20522983e-09,  2.38461667e-02,
                           -2.33540672e-09, 1.74823515e-10, -9.94345634e-03,
                           -6.14782246e-09,  6.09452763e-09, -1.39291346e-02],
                         [-1.16340880e+00, -5.68004778e-01, -2.18254923e-09,
                          1.21307403e+00, 6.64576848e-01,  1.49260050e-09,
                          -4.97333957e-02, -9.65238352e-02, 7.74806492e-10],
                         [-7.61157775e-01, -6.42928219e-01,  1.90093517e-10,
                          6.64585946e-01, 6.09644197e-01,  8.72800888e-10,
                          9.65585874e-02,  3.32486321e-02, -1.08771472e-09],
                         [-8.90246292e-10, -1.06078463e-10, -1.02164725e-02,
                          1.50223901e-09, 8.69409685e-10,  4.09774275e-02,
                          -7.02197099e-10, -8.40313571e-10, -3.07468472e-02],
                         [-1.14872616e+00,  5.52306917e-01, -6.27166936e-09,
                          -4.97336713e-02, 9.66013927e-02, -8.09399241e-10,
                          1.19832228e+00, -6.48950952e-01, 6.85905861e-09],
                         [7.45507209e-01, -6.33561850e-01,  6.12671503e-09,
                           -9.64812852e-02, 3.32487080e-02, -7.74615353e-10,
                           -6.48960123e-01, 6.00277064e-01, -5.08741780e-09],
                         [-7.51451327e-09,  6.14650239e-09, -1.41604277e-02,
                          6.62204039e-10, -1.02974704e-09, -3.07470051e-02,
                          6.85803822e-09, -5.09659842e-09, 4.49197416e-02]])

h2o_coords = np.array([[-0.0011, 0.3631, -0.0],
                       [-0.825, -0.1819, -0.0],
                       [0.8261, -0.1812, 0.0]])


# Ha/a0^2
co2_hessian_arr = np.array([[1.1314383525E+00, 4.2385767412E-04, 3.5051771425E-04, - 1.0501086627E+00, - 3.7813825173E-04,-3.4457384398E-04,  -8.1229733239E-02, -3.7456312285E-05, -6.5999542510E-05],
                            [4.2325160632E-04, 3.6570663096E-02, 1.2516781525E-07, - 3.8221942577E-04, - 7.3247574779E-02,-1.2460642412E-07,  -4.1000250417E-05,   3.6660190261E-02,  -2.4295086828E-08],
                            [3.4996749671E-04, 1.2517409883E-07, 3.6556726341E-02, - 2.8455375094E-04, - 9.9360220558E-08,-7.3235662640E-02,  -6.5386128105E-05,  -2.2911454518E-08,   3.6651071695E-02],
                            [-1.0457642155E+00, -3.8238813778E-04, -2.8469599658E-04, 2.0913922620E+00, 7.8828445803E-04,6.8225384914E-04,  -1.0458280713E+00,  -4.2242186106E-04,  -2.7748424189E-04],
                            [-3.7766992874E-04, -7.3411408385E-02, - 1.0641198331E-07, 7.8820556787E-04, 1.4684935286E-01,2.6867393467E-07, -4.1059908388E-04,  -7.3404560780E-02,  -1.1443188980E-07],
                            [-3.4413274803E-04, -1.3166345414E-07, - 7.3399496875E-02, 6.8218586806E-04, 2.6867411699E-07,1.4684780089E-01,  -3.3810770450E-04,  -1.4275024213E-07,  -7.3392638046E-02],
                            [-8.1229732348E-02, -4.1002276652E-05, - 6.5388156506E-05, - 1.0501723903E+00, - 4.1114029467E-04,-3.3853608979E-04,   1.1315021908E+00,  4.6040543323E-04,   3.4390672878E-04],
                            [-3.7453739804E-05, 3.6660190178E-02, - 2.2915437653E-08, - 4.2226004550E-04, - 7.3240736078E-02,-1.3510830784E-07,   4.5974573122E-04,  3.6563884025E-02,   1.3393048024E-07],
                            [-6.5997594969E-05, -2.4306709511E-08, 3.6651071540E-02, - 2.7734157097E-04, - 1.0679186055E-07,-7.3228811671E-02,   3.4336655361E-04,   1.3393871147E-02,  3.6549939293E-02]])


def assert_correct_co2_frequencies(hessian):
    """Ensure the projected frequencies of CO2 are roughly right"""

    assert sum(freq == 0.0 for freq in hessian.frequencies_proj) == 5

    # Should have a degenerate bending mode for CO2 with ν = 666 cm-1
    assert sum(np.isclose(Frequency(666, units='cm-1'), freq, atol=2.0)
               for freq in hessian.frequencies_proj) == 2

    # and two others that are larger
    assert sum(freq > Frequency(1000, units='cm-1')
               for freq in hessian.frequencies_proj) == 2


def test_hessian_set():

    h2o = Molecule(smiles='O')

    # Cannot set the Hessian as a matrix that isn't 3Nx3N
    with pytest.raises(ValueError):
        h2o.hessian = np.array([])

    with pytest.raises(ValueError):
        h2o.hessian = np.arange(0, 3*h2o.n_atoms)

    assert h2o.hessian is None
    h2o.hessian = np.zeros(shape=(3*h2o.n_atoms, 3*h2o.n_atoms))
    assert h2o.hessian is not None


def test_hessian_freqs():

    h2o = Molecule(smiles='O')
    h2o.hessian = h2o_hessian_arr

    assert isinstance(h2o.hessian, Hessian)
    freqs = h2o.hessian.frequencies

    # Should have 2 frequencies in the 3500 cm-1 range for the O-H stretches
    assert sum([Frequency(3000, units=wavenumber) < freq < Frequency(4000, units=wavenumber)
                for freq in freqs]) == 2

    # without projection there is an imaginary frequency
    assert sum(freq.is_imaginary for freq in freqs) == 1

    assert h2o.hessian.normal_modes[0].shape == (h2o.n_atoms, 3)

    # Projecting should give frequencies close to those obtained from ORCA
    # the vibrational frequencies are the largest three (non-zero)
    nu_1, nu_2, nu_3 = h2o.hessian.frequencies_proj[-3:]
    assert np.isclose(nu_1, 1567.610851, atol=1.0)
    assert np.isclose(nu_2, 3467.698182, atol=1.0)
    assert np.isclose(nu_3, 3651.462209, atol=1.0)


def test_hessian_modes():
    """Ensure the translational, rotational and vibrational modes are close
    to the expected values for a projected Hessian"""

    h2o = Molecule(smiles='O')
    h2o.coordinates = h2o_coords
    h2o.hessian = h2o_hessian_arr

    for trans_mode in h2o.hessian.normal_modes_proj[:3]:
        assert np.allclose(trans_mode, np.zeros(shape=(h2o.n_atoms, 3)))

    for rot_mode in h2o.hessian.normal_modes_proj[3:6]:
        assert np.allclose(rot_mode, np.zeros(shape=(h2o.n_atoms, 3)))

    for vib_mode in h2o.hessian.normal_modes_proj[6:]:

        # Vibrational modes should have no component in the z-axis
        for i, _ in enumerate(h2o.atoms):
            assert np.isclose(vib_mode[i, 2], 0.0, atol=1E-4)


def test_hessian_linear_freqs():

    co2 = Molecule('CO2_opt.xyz')
    assert co2.is_linear()

    co2.hessian = Hessian(co2_hessian_arr, units='Ha/a0^2')
    assert_correct_co2_frequencies(hessian=co2.hessian)


def test_gaussian_hessian_extract_h2():

    h2 = ade.Molecule(atoms=[ade.Atom('H', x=0.3804),
                             ade.Atom('H', x=-0.3804)])

    calc = Calculation(name='tmp',
                       molecule=h2,
                       method=ade.methods.G09(),
                       keywords=ade.HessianKeywords())

    calc.output.filename = 'H2_hess_g09.log'
    h2.hessian = calc.get_hessian()

    assert np.isclose(h2.hessian.frequencies[-1], Frequency(4383.9811),
                      atol=1.0)

    assert np.isclose(h2.hessian.frequencies_proj[-1], Frequency(4383.9811),
                      atol=1.0)


def test_gaussian_hessian_extract_co2():

    co2 = Molecule('CO2_opt.xyz')

    calc = Calculation(name='tmp',
                       molecule=co2,
                       method=ade.methods.G09(),
                       keywords=ade.SinglePointKeywords([]))

    calc.output.filename = 'CO2_opt_hess_g09.log'
    co2.hessian = calc.get_hessian()

    assert all(np.isclose(freq, Frequency(0, units='cm-1'), atol=10) for freq
               in co2.hessian.frequencies[:5])

    assert all(freq == 0.0 for freq in co2.hessian.frequencies_proj[:5])

    assert_correct_co2_frequencies(hessian=co2.hessian)


def test_nwchem_hessian_extract_h2o():

    calc = Calculation(name='tmp',
                       molecule=ade.Molecule(smiles='O'),
                       method=ade.methods.NWChem(),
                       keywords=ade.HessianKeywords())

    calc.output.filename = 'H2O_hess_nwchem.out'

    hessian = calc.get_hessian()

    for freqs in (hessian.frequencies, hessian.frequencies_proj):
        assert sum(np.isclose(freq, 0.0, atol=15) for freq in freqs) == 6

        assert sum(np.isclose(freq, Frequency(1642.78), atol=4)
                   for freq in freqs) == 1

        assert sum(np.isclose(freq, Frequency(3860.38), atol=4)
                   for freq in freqs) == 1

        assert sum(np.isclose(freq, Frequency(3959.20), atol=4)
                   for freq in freqs) == 1
