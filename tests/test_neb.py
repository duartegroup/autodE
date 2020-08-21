from autode.neb import neb
from autode.species.molecule import Molecule, Species
from autode.species.molecule import Reactant, Product
from autode.atoms import Atom
from autode.pes.pes import BreakingBond, FormingBond
from autode.input_output import xyz_file_to_atoms
from autode.methods import XTB
import shutil
import os

here = os.path.dirname(os.path.abspath(__file__))

data_path = os.path.join(here, 'data', 'neb')

init_xyz = os.path.join(data_path, 'sn2_init.xyz')
final_xyz = os.path.join(data_path, 'sn2_final.xyz')


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


def test_full_calc_with_xtb():

    sn2_neb = neb.NEB(initial_species=Species(name='inital', charge=-1, mult=0,
                                              atoms=xyz_file_to_atoms(init_xyz)),
                      final_species=Species(name='final', charge=-1, mult=0,
                                            atoms=xyz_file_to_atoms(final_xyz)),
                      num=14)

    os.chdir(data_path)
    sn2_neb.interpolate_geometries()

    xtb = XTB()

    # Don't run the NEB without a working XTB install
    if shutil.which('xtb') is None or  not shutil.which('xtb').endswith('xtb'):
        return

    xtb.path = shutil.which('xtb')
    sn2_neb.calculate(method=xtb, n_cores=2)

    # There should be a peak in this surface
    assert len(list(sn2_neb.get_species_saddle_point())) > 0

    assert all(image.energy is not None for image in sn2_neb.images)

    energies = [image.energy for image in sn2_neb.images]
    path_energy = sum(energy - min(energies) for energy in energies)

    assert 0.35 < path_energy < 0.45

    if os.path.exists('NEB'):
        shutil.rmtree('NEB')

    os.chdir(here)


def test_get_ts_guess_neb():

    reactant = Reactant(name='inital', charge=-1, mult=0, solvent_name='water',
                        atoms=xyz_file_to_atoms(init_xyz))

    product = Reactant(name='final', charge=-1, mult=0, solvent_name='water',
                       atoms=xyz_file_to_atoms(final_xyz))

    xtb = XTB()

    # Don't run the NEB without a working XTB install
    if shutil.which('xtb') is None or  not shutil.which('xtb').endswith('xtb'):
        return

    xtb.path = shutil.which('xtb')

    ts_guess = neb.get_ts_guess_neb(reactant, product,
                                    method=xtb, n=10)

    assert ts_guess is not None
    # Approximate distances at the TS guess
    assert 1.8 < ts_guess.get_distance(0, 2) < 2.2      # C-F
    assert 1.9 < ts_guess.get_distance(2, 1) < 2.3      # C-Cl

    if os.path.exists('NEB'):
        shutil.rmtree('NEB')

    if os.path.exists('neb.xyz'):
        os.remove('neb.xyz')
