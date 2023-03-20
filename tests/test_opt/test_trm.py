import os
import numpy as np
from autode.species import Molecule
from autode.atoms import Atom
from autode.methods import XTB
from autode.values import Gradient
from autode.hessians import Hessian
from autode.utils import work_in_tmp_dir
from ..test_utils import requires_with_working_xtb_install
from autode.opt.coordinates import CartesianCoordinates, DIC
from autode.opt.optimisers.trm import HybridTRMOptimiser


def test_trim_step():
    water_atoms = [
        Atom("O", -0.0011, 0.3631, -0.0000),
        Atom("H", -0.8250, -0.1819, -0.0000),
        Atom("H", 0.8261, -0.1812, 0.0000),
    ]
    water = Molecule(atoms=water_atoms)
    gradient = Gradient(
        [
            [-2.63152236e-03, 2.78810423e-02, -9.79654122e-17],
            [-4.98983473e-02, -1.31220306e-02, 7.52534291e-17],
            [5.25298697e-02, -1.47590118e-02, 2.27119831e-17],
        ],
        units="Ha/ang",
    )
    hessian = Hessian(
        np.array(
            [
                [
                    1.84603932e00,
                    8.88206242e-03,
                    1.17680208e-05,
                    -9.30189149e-01,
                    -5.97884652e-01,
                    5.49656836e-04,
                    -9.15908584e-01,
                    5.88934444e-01,
                    -5.37432181e-04,
                ],
                [
                    8.88206242e-03,
                    1.12859077e00,
                    9.90019876e-04,
                    -4.63105513e-01,
                    -5.69028469e-01,
                    3.90496797e-04,
                    4.44924224e-01,
                    -5.60586801e-01,
                    3.80243116e-04,
                ],
                [
                    1.17680208e-05,
                    9.90019876e-04,
                    5.12000041e-02,
                    -4.23800204e-04,
                    -5.00058419e-04,
                    -5.91796882e-02,
                    4.12032183e-04,
                    -4.89961457e-04,
                    -6.22312785e-02,
                ],
                [
                    -9.30189149e-01,
                    -4.63105513e-01,
                    -4.23800204e-04,
                    1.02121421e00,
                    5.34296137e-01,
                    -5.38584660e-04,
                    -8.53582796e-02,
                    -6.86591609e-02,
                    1.10777160e-05,
                ],
                [
                    -5.97884652e-01,
                    -5.69028469e-01,
                    -5.00058419e-04,
                    5.34296137e-01,
                    5.13466077e-01,
                    -3.77066841e-04,
                    6.82855066e-02,
                    5.60825165e-02,
                    -1.33269486e-05,
                ],
                [
                    5.49656836e-04,
                    3.90496797e-04,
                    -5.91796882e-02,
                    -5.38584660e-04,
                    -3.77066841e-04,
                    4.22748644e-02,
                    -1.10721757e-05,
                    -1.34299563e-05,
                    5.19934921e-02,
                ],
                [
                    -9.15908584e-01,
                    4.44924224e-01,
                    4.12032183e-04,
                    -8.53582796e-02,
                    6.82855066e-02,
                    -1.10721757e-05,
                    9.95658499e-01,
                    -5.15673048e-01,
                    5.26354465e-04,
                ],
                [
                    5.88934444e-01,
                    -5.60586801e-01,
                    -4.89961457e-04,
                    -6.86591609e-02,
                    5.60825165e-02,
                    -1.34299563e-05,
                    -5.15673048e-01,
                    5.05008656e-01,
                    -3.66916168e-04,
                ],
                [
                    -5.37432181e-04,
                    3.80243116e-04,
                    -6.22312785e-02,
                    1.10777160e-05,
                    -1.33269486e-05,
                    5.19934921e-02,
                    5.26354465e-04,
                    -3.66916168e-04,
                    4.53600807e-02,
                ],
            ]
        )
    )

    opt = HybridTRMOptimiser(coord_type="dic", maxiter=2, gtol=0.1, etol=0.1)
    opt._species = water
    opt._build_coordinates()
    assert isinstance(opt._coords, DIC)
    opt._coords.update_g_from_cart_g(gradient)
    opt._coords.update_h_from_cart_h(hessian)

    h_eff = opt._get_trm_minimise_h_eff()
    step = -np.linalg.inv(h_eff) @ opt._coords.g
    new_coords = opt._coords + step
    cart_step = new_coords.to("cart") - opt._coords.to("cart")
    step_size = np.linalg.norm(cart_step)

    assert np.isclose(step_size, opt.alpha, rtol=0.01)  # 1% error margin

    opt = HybridTRMOptimiser(coord_type="cart", maxiter=2, gtol=0.1, etol=0.1)
    opt._species = water
    opt._build_coordinates()
    assert isinstance(opt._coords, CartesianCoordinates)
    opt._coords.update_g_from_cart_g(gradient)
    opt._coords.update_h_from_cart_h(hessian)

    h_eff = opt._get_trm_minimise_h_eff()
    step = -np.linalg.inv(h_eff) @ opt._coords.g  # in Cartesian
    step_size = np.linalg.norm(step)

    # for Cartesian, the step size should be 0.1% of trust radius
    assert np.isclose(step_size, opt.alpha, rtol=0.001)


def test_damping_in_hybridtrim_optimiser():
    coord1 = CartesianCoordinates([1.0, -2.0, 1.0, 3.0, 1.1, 1.2])
    coord1.e = 0.14
    coord1.g = np.array([0.2, 0.3, 0.1, -0.2])
    coord2 = CartesianCoordinates([1.1, -1.9, 1.1, 3.1, 1.2, 1.3])
    coord2.e = 0.10
    coord2.g = np.array([0.1, 0.3, 0.01, -0.1])
    opt = HybridTRMOptimiser(coord_type="cart", maxiter=2, gtol=0.1, etol=0.1)

    # simulate an oscillation happening
    opt._coords = coord1
    opt._coords = coord2
    opt._coords = coord1
    opt._coords = coord2

    assert opt._damp_if_required()  # should damp
    # 50% mixing
    avg_coord = (coord1 + coord2) / 2.0
    assert np.allclose(avg_coord, opt._coords)


@work_in_tmp_dir()
@requires_with_working_xtb_install
def test_trim_molecular_opt():

    mol = Molecule(smiles="O")
    assert [atom.label for atom in mol.atoms] == ["O", "H", "H"]

    HybridTRMOptimiser.optimise(mol, method=XTB())

    # Check optimised distances are similar to running the optimiser in XTB
    for oh_atom_idx_pair in [(0, 1), (0, 2)]:
        assert np.isclose(
            mol.distance(*oh_atom_idx_pair).to("Å"), 0.9595, atol=1e-2
        )

    assert np.isclose(mol.distance(1, 2), 1.5438, atol=1e-2)


@work_in_tmp_dir()
@requires_with_working_xtb_install
def test_optimiser_plotting():
    mol = Molecule(smiles="O")

    opt = HybridTRMOptimiser(maxiter=100, gtol=1e-3, etol=1e-3)
    opt.run(mol, method=XTB())

    opt.plot_optimisation()
    assert os.path.isfile(f"{mol.name}_opt_plot.pdf")
