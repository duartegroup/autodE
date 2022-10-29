from autode.species.complex import NCIComplex
from autode.species.molecule import Molecule
from autode.config import Config
from autode import geom


def test_nci_complex():

    water = Molecule(name="water", smiles="O")
    f = Molecule(name="formaldehyde", smiles="C=O")

    nci_complex = NCIComplex(f, water)
    assert nci_complex.n_atoms == 7

    # Set so the number of conformers doesn't explode
    Config.num_complex_sphere_points = 6
    Config.num_complex_random_rotations = 4

    nci_complex._generate_conformers()
    assert len(nci_complex.conformers) == 24

    for conformer in nci_complex.conformers:
        # conformer.print_xyz_file()
        assert geom.are_coords_reasonable(coords=conformer.coordinates)

    # To view the structures generated overlaid
    # --------------------------------------------------

    # from autode.input_output import atoms_to_xyz_file
    # all_atoms = []
    # for c in nci_complex.conformers:
    #     all_atoms += c.atoms
    # atoms_to_xyz_file(atoms=all_atoms, filename='tmp.xyz')
