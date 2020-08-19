from autode.neb import neb
from autode.species.molecule import Molecule, Species
from autode.atoms import Atom
from autode.pes.pes import BreakingBond, FormingBond
from autode.input_output import xyz_file_to_atoms
import os

here = os.path.dirname(os.path.abspath(__file__))


def test_removing_active_bonds():

    filepath = os.path.join(here, 'data', 'neb', 'co_complex.xyz')

    mol = Molecule(name='co_complex',
                   atoms=xyz_file_to_atoms(filepath))

    bbond = BreakingBond(atom_indexes=(1, 2), species=mol)
    fbond = FormingBond(atom_indexes=(3, 1), species=mol)

    bonds = neb.active_bonds_no_rings(mol, fbonds=[fbond], bbonds=[bbond])

    # Should remove the breaking bond as it forms a ring with an already
    # present atom pair
    assert len(bonds) == 1
    assert isinstance(bonds[0], FormingBond)

    # Also check that the number of images to generate are somewhat reasonable
    n_images = neb.calc_n_images(fbonds=[fbond], bbonds=[bbond],
                                 average_spacing=0.1)
    assert 0 < n_images < 20
    assert n_images % 2 == 0


def test_contains_peak():

    species_list = []
    for i in range(5):
        h2 = Species(name='h2', charge=0, mult=2,
                     atoms=[Atom('H'), Atom('H', x=0)])

        h2.energy = i
        species_list.append(h2)

    assert not neb.contains_peak(species_list)

    species_list[2].energy = 5
    assert neb.contains_peak(species_list)
