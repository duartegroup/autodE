import numpy as np
from autode import Molecule
from autode.atoms import Atom
from autode.ext.ade_rb_opt import opt_rb_coords
from autode.geom import are_coords_reasonable


def rb_minimised_is_reasonable(molecule):
    n_atoms = molecule.n_atoms
    coords = opt_rb_coords(
        py_coords=molecule.coordinates,
        py_bonded_matrix=molecule.bond_matrix,
        py_r0_matrix=np.asarray(
            molecule.graph.eqm_bond_distance_matrix, dtype="f8"
        ),
        py_k_matrix=1 * np.ones((n_atoms, n_atoms), dtype="f8"),
        py_c_matrix=0.01 * np.ones((n_atoms, n_atoms), dtype="f8"),
        py_exponent=8,
    )

    molecule.coordinates = coords
    molecule.print_xyz_file(filename="tmp.xyz")

    assert are_coords_reasonable(coords)


def test_h2():
    """Simple 1D optimisation of H2, slightly displaced from its minimum"""

    h2 = Molecule(atoms=[Atom("H"), Atom("H", x=1.1)])

    # Add a bond between the two H atoms
    h2.graph.add_edge(0, 1)

    rb_minimised_is_reasonable(h2)
    assert np.isclose(h2.distance(0, 1), 0.8, atol=0.1)


def test_alkanes():
    rb_minimised_is_reasonable(molecule=Molecule(smiles="C"))
    rb_minimised_is_reasonable(molecule=Molecule(smiles="CCCC"))


def _test_tmp():
    butane = Molecule(smiles="CCCCCCCC")

    from autode.conformers.conf_gen import get_simanl_conformer

    tmp = get_simanl_conformer(butane, save_xyz=False)
    tmp.print_xyz_file(filename="tmp.xyz")

    # butane.populate_conformers(n_confs=5)
    # for i, conf in enumerate(butane.conformers):
    #     conf.print_xyz_file(filename=f'tmp{i}.xyz')
