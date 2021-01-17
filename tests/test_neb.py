from autode.neb import neb
from autode.mol_graphs import find_cycles
from autode.species.molecule import Molecule, Species
from autode.species.molecule import Reactant, Product
from autode.atoms import Atom
from autode.pes.pes import BreakingBond, FormingBond
from autode.input_output import xyz_file_to_atoms
from autode.methods import XTB
from . import testutils
import shutil
import os

here = os.path.dirname(os.path.abspath(__file__))


@testutils.work_in_zipped_dir(os.path.join(here, 'data', 'neb.zip'))
def test_removing_active_bonds():

    mol = Molecule(name='co_complex',
                   atoms=xyz_file_to_atoms('co_complex.xyz'))

    bbond = BreakingBond(atom_indexes=(1, 2), species=mol)
    fbond = FormingBond(atom_indexes=(3, 1), species=mol)

    bonds = neb.active_bonds_no_rings(mol, fbonds=[fbond], bbonds=[bbond])

    # Should remove the breaking bond as it forms a ring with an already
    # present atom pair
    assert len(bonds) == 1
    assert isinstance(bonds[0], FormingBond)

    # If there's only the breaking bond then it should not be removed
    bonds = neb.active_bonds_no_rings(mol, bbonds=[bbond], fbonds=[])
    assert len(bonds) == 1

    # Also check that the number of images to generate are somewhat reasonable
    n_images = neb.calc_n_images(fbonds=[fbond], bbonds=[bbond],
                                 average_spacing=0.1)
    assert 0 < n_images < 20


def test_removing_active_bonds2():

    mol = Molecule(name='oxirane_OAc',
                   atoms=[Atom('O',  2.67572,   0.54993, -2.85839),
                          Atom('C',  3.38360,   0.73831, -4.01669),
                          Atom('C',  4.84747,   0.86062, -4.11723),
                          Atom('O',  2.73479,   0.81467, -5.09355),
                          Atom('H',  5.26716,   0.15148, -4.88617),
                          Atom('H',  5.12702,   1.87661, -4.49574),
                          Atom('H',  5.38360,   0.64649, -3.18798),
                          Atom('C',  0.75870,   0.10430, -0.01960),
                          Atom('C', -0.75300,  -0.09360,  0.00430),
                          Atom('O',  0.01850, -0.08750,  1.13910),
                          Atom('H',  1.36090, -0.77540, -0.3192),
                          Atom('H',  1.13650,   1.12180, -0.2675),
                          Atom('H', -1.11720, -1.06370, -0.4253),
                          Atom('H', -1.40430,  0.79400, -0.11170)])

    assert (7, 9) in mol.graph.edges

    rings = find_cycles(mol.graph)
    # Check that the rings have been found correctly
    assert (any(7 in ring for ring in rings)
            and any(9 in ring for ring in rings))

    # O attack on C, breaking the C-O in the 3-membered ring
    bbond = BreakingBond(atom_indexes=(7, 9), species=mol)
    fbond = FormingBond(atom_indexes=(0, 7), species=mol)

    bonds = neb.active_bonds_no_rings(mol, fbonds=[fbond], bbonds=[bbond])
    assert len(bonds) == 1
    assert isinstance(bonds[0], FormingBond)


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

    species_list[2].energy = None
    assert not neb.contains_peak(species_list)


@testutils.work_in_zipped_dir(os.path.join(here, 'data', 'neb.zip'))
def test_full_calc_with_xtb():

    sn2_neb = neb.NEB(initial_species=Species(name='inital', charge=-1, mult=0,
                                              atoms=xyz_file_to_atoms('sn2_init.xyz'),
                                              solvent_name='water'),
                      final_species=Species(name='final', charge=-1, mult=0,
                                            atoms=xyz_file_to_atoms('sn2_final.xyz'),
                                            solvent_name='water'),
                      num=14)

    sn2_neb.interpolate_geometries()

    xtb = XTB()

    # Don't run the NEB without a working XTB install
    if shutil.which('xtb') is None or not shutil.which('xtb').endswith('xtb'):
        return

    xtb.path = shutil.which('xtb')
    sn2_neb.calculate(method=xtb, n_cores=2)

    # There should be a peak in this surface
    peak_species = sn2_neb.get_species_saddle_point()
    assert peak_species is not None

    assert all(image.energy is not None for image in sn2_neb.images)

    energies = [image.energy for image in sn2_neb.images]
    path_energy = sum(energy - min(energies) for energy in energies)

    assert 0.25 < path_energy < 0.45


@testutils.work_in_zipped_dir(os.path.join(here, 'data', 'neb.zip'))
def test_get_ts_guess_neb():

    reactant = Reactant(name='inital', charge=-1, mult=0, solvent_name='water',
                        atoms=xyz_file_to_atoms('sn2_init.xyz'))

    product = Reactant(name='final', charge=-1, mult=0, solvent_name='water',
                       atoms=xyz_file_to_atoms('sn2_final.xyz'))

    xtb = XTB()

    # Don't run the NEB without a working XTB install
    if shutil.which('xtb') is None or  not shutil.which('xtb').endswith('xtb'):
        return

    xtb.path = shutil.which('xtb')

    ts_guess = neb.get_ts_guess_neb(reactant, product,
                                    method=xtb, n=10)

    assert ts_guess is not None
    # Approximate distances at the TS guess
    assert 1.8 < ts_guess.distance(0, 2) < 2.3      # C-F
    assert 1.9 < ts_guess.distance(2, 1) < 2.5      # C-Cl

    if os.path.exists('NEB'):
        shutil.rmtree('NEB')

    if os.path.exists('neb.xyz'):
        os.remove('neb.xyz')
