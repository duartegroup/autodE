import shutil
import os
from autode.path import Path
from autode.neb import NEB, CINEB
from autode.neb.ci import Images, CImages, Image
from autode.species.molecule import Species
from autode.species.molecule import Reactant
from autode.neb.neb import get_ts_guess_neb
from autode.atoms import Atom
from autode.input_output import xyz_file_to_atoms
from autode.methods import XTB, ORCA
from . import testutils


here = os.path.dirname(os.path.abspath(__file__))


def test_neb_properties():

    # H-H  H
    reac = Species(name='reac', charge=0, mult=2,
                   atoms=[Atom('H'), Atom('H', x=0.7), Atom('H', x=2.0)])
    # H  H-H
    prod = Species(name='prod', charge=0, mult=2,
                   atoms=[Atom('H'), Atom('H', x=1.3), Atom('H', x=2.0)])

    neb = NEB(initial_species=reac, final_species=prod, num=3)
    assert len(neb.images) == 3
    assert neb.get_species_saddle_point() is None
    assert not neb.contains_peak()

    neb.partition(n=2)
    assert len(neb.images) == 5

    # Should move monotonically from 0.7 -> 1.3 Angstroms
    for i in range(1, 5):

        prev_bb_dist = neb.images[i-1].species.distance(0, 1)
        curr_bb_dist = neb.images[i].species.distance(0, 1)

        assert curr_bb_dist > prev_bb_dist

    if os.path.exists('neb.xyz'):
        os.remove('neb.xyz')


def test_image_properties():

    images = CImages(images=[Image(name='tmp', k=0.1)])
    assert images != 0
    assert images == images

    images = Images(init_k=0.1, num=1)
    assert images != 0
    assert images == images


def test_contains_peak():

    species_list = Path()
    for i in range(5):
        h2 = Species(name='h2', charge=0, mult=2,
                     atoms=[Atom('H'), Atom('H', x=0)])

        h2.energy = i
        species_list.append(h2)

    assert not species_list.contains_peak

    species_list[2].energy = 5
    assert species_list.contains_peak

    species_list[2].energies.clear()
    species_list[2].energy = None
    assert not species_list.contains_peak


@testutils.work_in_zipped_dir(os.path.join(here, 'data', 'neb.zip'))
def test_full_calc_with_xtb():

    sn2_neb = NEB(initial_species=Species(name='inital', charge=-1, mult=0,
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
    if shutil.which('xtb') is None or not shutil.which('xtb').endswith('xtb'):
        return

    xtb.path = shutil.which('xtb')

    ts_guess = get_ts_guess_neb(reactant, product, method=xtb, n=10)

    assert ts_guess is not None
    # Approximate distances at the TS guess
    assert 1.8 < ts_guess.distance(0, 2) < 2.3      # C-F
    assert 1.9 < ts_guess.distance(2, 1) < 2.5      # C-Cl

    if os.path.exists('NEB'):
        shutil.rmtree('NEB')

    if os.path.exists('neb.xyz'):
        os.remove('neb.xyz')

    # Trying to get a TS guess with an unavailable method should return None
    # as a TS guess
    orca = ORCA()
    orca.path = None

    orca_ts_guess = get_ts_guess_neb(reactant, product, method=orca, n=10)
    assert orca_ts_guess is None


def test_climbing_image():

    images = CImages(images=[Image(name='tmp', k=0.1)])
    assert images.peak_idx is None
    assert images[0].iteration == 0
    images[0].iteration = 10

    images.increment()
    # with no peak there should be no Image -> CIImage change
    assert images[0].iteration == 10
    assert isinstance(images[0], Image)
