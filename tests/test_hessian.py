import pytest
import numpy as np
from autode.species import Molecule
from autode.values import Frequency
from autode.thermo.hessians import Hessian
from autode.units import wavenumber


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
    h2o.hessian = np.array([[2.31423829e+00,  1.56166837e-02,  8.61890193e-09,
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

    print(h2o.frequencies)

    assert isinstance(h2o.hessian, Hessian)
    freqs = h2o.hessian.frequencies

    # Should have 2 frequencies in the 3500 cm-1 range for the O-H stretches
    assert sum([Frequency(3000, units=wavenumber) < freq < Frequency(4000, units=wavenumber)
                for freq in freqs]) == 2

    # without projection there is an imaginary frequency
    assert sum(freq.is_imaginary for freq in freqs) == 1

    assert h2o.hessian.normal_modes[0].shape == (h2o.n_atoms, 3)
    print(h2o.hessian.normal_modes[0])
