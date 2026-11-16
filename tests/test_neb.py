import shutil
import os
import numpy as np
import pytest
from autode.path import Path
from autode.neb import NEB
from autode.values import Distance, ForceConstant
from autode.neb.ci import Images, CImages, Image
from autode.neb.idpp import IDPP
from autode.species.molecule import Species, Molecule
from autode.species.molecule import Reactant
from autode.neb.neb import get_ts_guess_neb
from autode.neb.original import energy_gradient
from autode.atoms import Atom
from autode.geom import are_coords_reasonable, calc_rmsd
from autode.input_output import xyz_file_to_atoms
from autode.utils import work_in_tmp_dir, work_in
from autode.methods import XTB, ORCA
from . import testutils
from .testutils import work_in_zipped_dir

here = os.path.dirname(os.path.abspath(__file__))


@work_in_tmp_dir()
def test_neb_properties():
    # H-H  H
    reac = Species(
        name="reac",
        charge=0,
        mult=2,
        atoms=[Atom("H"), Atom("H", x=0.7), Atom("H", x=2.0)],
    )
    # H  H-H
    prod = Species(
        name="prod",
        charge=0,
        mult=2,
        atoms=[Atom("H"), Atom("H", x=1.3), Atom("H", x=2.0)],
    )

    neb = NEB.from_end_points(reac, prod, num=3)
    assert len(neb.images) == 3
    assert neb.peak_species is None
    assert not neb.images.contains_peak

    # Should move monotonically from 0.7 -> 1.3 Angstroms
    for i in range(1, len(neb.images)):
        prev_bb_dist = neb.images[i - 1].distance(0, 1)
        curr_bb_dist = neb.images[i].distance(0, 1)

        assert curr_bb_dist > prev_bb_dist


def test_image_properties():
    k = ForceConstant(0.1)
    images = CImages(images=Images(init_k=k))
    assert images != 0
    assert images == images

    images = Images(init_k=k)
    assert images != 0
    assert images == images

    image = Image(Molecule(smiles="CC"), k=ForceConstant(1.0), name="tmp")
    with pytest.raises(Exception):
        image._generate_conformers()


def test_contains_peak():
    species_list = Path()
    for i in range(5):
        h2 = Species(
            name="h2", charge=0, mult=2, atoms=[Atom("H"), Atom("H", x=0)]
        )

        h2.energy = i
        species_list.append(h2)

    assert not species_list.contains_peak

    species_list[2].energy = 5
    assert species_list.contains_peak

    species_list[2].energies.clear()
    species_list[2].energy = None
    assert not species_list.contains_peak


@testutils.requires_working_xtb_install
@testutils.work_in_zipped_dir(os.path.join(here, "data", "neb.zip"))
def test_full_calc_with_xtb():
    sn2_neb = NEB.from_end_points(
        initial=Species(
            name="inital",
            charge=-1,
            mult=1,
            atoms=xyz_file_to_atoms("sn2_init.xyz"),
            solvent_name="water",
        ),
        final=Species(
            name="final",
            charge=-1,
            mult=1,
            atoms=xyz_file_to_atoms("sn2_final.xyz"),
            solvent_name="water",
        ),
        num=14,
    )

    sn2_neb.calculate(method=XTB(), n_cores=2)

    # There should be a peak in this surface
    assert sn2_neb.peak_species is not None

    assert all(image.energy is not None for image in sn2_neb.images)

    energies = [image.energy for image in sn2_neb.images]
    path_energy = sum(energy - min(energies) for energy in energies)

    assert 0.25 < path_energy < 0.45


@testutils.requires_working_xtb_install
@testutils.work_in_zipped_dir(os.path.join(here, "data", "neb.zip"))
def test_get_ts_guess_neb():
    reactant = Reactant(
        name="inital",
        charge=-1,
        mult=1,
        solvent_name="water",
        atoms=xyz_file_to_atoms("sn2_init.xyz"),
    )

    product = Reactant(
        name="final",
        charge=-1,
        mult=1,
        solvent_name="water",
        atoms=xyz_file_to_atoms("sn2_final.xyz"),
    )

    xtb = XTB()
    xtb.path = shutil.which("xtb")

    ts_guess = get_ts_guess_neb(reactant, product, method=xtb, n=10)

    assert ts_guess is not None
    # Approximate distances at the TS guess
    assert 1.8 < ts_guess.distance(0, 2) < 2.3  # C-F
    assert 2.1 < ts_guess.distance(2, 1) < 2.6  # C-Cl

    if os.path.exists("NEB"):
        shutil.rmtree("NEB")

    if os.path.exists("neb.xyz"):
        os.remove("neb.xyz")

    # Trying to get a TS guess with an unavailable method should return None
    # as a TS guess
    orca = ORCA()
    orca.path = None

    orca_ts_guess = get_ts_guess_neb(reactant, product, method=orca, n=10)
    assert orca_ts_guess is None


def test_climbing_image():
    k = ForceConstant(0.1)
    images = CImages(images=Images(init_k=k))
    images.append_species(Molecule(atoms=[Atom("H")], mult=2))

    assert images.peak_idx is None
    assert images[0].iteration == 0
    images[0].iteration = 10


def _simple_h2_images(num, shift, increment):
    """Simple set of images for a n-image NEB for H2"""

    images = Images(init_k=ForceConstant(1.0))

    for i in range(num):
        mol = Molecule(atoms=[Atom("H"), Atom("H", x=shift + i * increment)])
        images.append_species(mol)

    return images


def test_energy_gradient_type():
    k = ForceConstant(1.0)
    image = Image(species=Molecule(atoms=[Atom("H")], mult=2), name="tmp", k=k)

    # Energy and gradient must have a method (EST or IDPP)
    with pytest.raises(AssertionError):
        _ = energy_gradient(image=image, method=None, n_cores=1)


def test_iddp_init():
    """IDPP requires at least 3 images"""
    with pytest.raises(AssertionError):
        _ = IDPP(n_images=2)

    images = _simple_h2_images(num=2, shift=0.5, increment=0.1)
    coords1 = images[0].coordinates
    coords2 = images[1].coordinates

    # k_spr has to be positive
    idpp = IDPP(n_images=4, k_spr=-1.0)
    with pytest.raises(RuntimeError):
        idpp.get_path(coords1, coords2)

    # rms gtol has to be positive
    idpp = IDPP(n_images=4, rms_gtol=-1.0)
    with pytest.raises(RuntimeError):
        idpp.get_path(coords1, coords2)

    # masxiter has to be negative
    idpp = IDPP(n_images=4, maxiter=-10)
    with pytest.raises(RuntimeError):
        idpp.get_path(coords1, coords2)


@work_in_zipped_dir(os.path.join(here, "data", "neb.zip"))
def test_sequential_idpp():
    rct = Molecule("ag-cycl-rct.xyz")
    prod = Molecule("ag-cycl-prod.xyz")
    neb1 = NEB.from_end_points(rct, prod, num=10, sidpp=True)
    neb2 = NEB.from_end_points(rct, prod, num=10, sidpp=False)
    # should be essentially the same geometry in both cases
    for i in range(len(neb1.images)):
        assert (
            calc_rmsd(neb1.images[i].coordinates, neb2.images[i].coordinates)
            < 0.1
        )


@work_in_zipped_dir(os.path.join(here, "data", "neb.zip"))
def test_idpp_relaxing_a_path():
    # relax a path manually
    rct = Molecule("ag-cycl-rct.xyz")
    prod = Molecule("ag-cycl-prod.xyz")
    interp_species = []
    for i in range(1, 11):
        tmp_spc = rct.new_species(name=f"interp_{i}")
        tmp_spc.coordinates = rct.coordinates + (i / 11) * (
            prod.coordinates - rct.coordinates
        )
        interp_species.append(tmp_spc)

    neb1 = NEB.from_list([rct, *interp_species, prod])
    neb1.idpp_relax()

    neb2 = NEB.from_end_points(rct, prod, num=12, sidpp=False)
    assert len(neb1.images) == len(neb2.images)

    for i in range(len(neb1.images)):
        assert (
            calc_rmsd(neb1.images[i].coordinates, neb2.images[i].coordinates)
            < 0.1
        )


@work_in_zipped_dir(os.path.join(here, "data", "neb.zip"))
def test_idpp_path_length():
    rct = Molecule("ag-cycl-rct.xyz")
    prod = Molecule("ag-cycl-prod.xyz")
    neb = NEB.from_end_points(rct, prod, num=10, sidpp=True)
    assert len(neb.images) == 10

    path_l = 0.0
    for i in range(1, len(neb.images)):
        path_l += np.linalg.norm(
            neb.images[i - 1].coordinates - neb.images[i].coordinates
        )
    assert path_l > 0.0
    idpp = IDPP(n_images=10, sequential=True)
    path_cpp = idpp.get_path_length(rct.coordinates, prod.coordinates)
    assert np.isclose(path_l, path_cpp)


@work_in_tmp_dir()
def test_neb_interpolate_and_idpp_relax():
    mol = Molecule(
        name="methane",
        atoms=[
            Atom("C", -0.91668, 0.42765, 0.00000),
            Atom("H", 0.15332, 0.42765, 0.00000),
            Atom("H", -1.27334, 0.01569, -0.92086),
            Atom("H", -1.27334, 1.43112, 0.10366),
            Atom("H", -1.27334, -0.16385, 0.81720),
        ],
    )

    rot_mol = mol.copy()
    rot_mol.rotate(axis=[1.0, 0.0, 0.0], theta=1.5)

    neb = NEB.from_end_points(initial=mol, final=rot_mol, num=10)

    for image in neb.images:
        assert are_coords_reasonable(image.coordinates)


def test_max_delta_between_images():
    _list = [
        Molecule(atoms=[Atom("H"), Atom("H", x=2.7)]),
        Molecule(atoms=[Atom("H"), Atom("H", x=1.7)]),
    ]

    assert np.isclose(
        NEB.from_list(_list).max_atom_distance_between_images, 1.0
    )

    _list[0].atoms[1].coord[0] = 1.7  # x coordinate of the second atom
    assert np.isclose(
        NEB.from_list(_list).max_atom_distance_between_images, 0.0
    )


def test_max_delta_between_images_h3():
    _list = [
        Molecule(atoms=[Atom("H"), Atom("H", x=0.7), Atom("H", x=2.7)]),
        Molecule(atoms=[Atom("H"), Atom("H", x=0.70657), Atom("H", x=2.7)]),
    ]

    neb = NEB.from_list(_list)
    assert np.isclose(neb.max_atom_distance_between_images, 0.00657)

    assert np.isclose(
        neb.max_atom_distance_between_images,
        neb._max_atom_distance_between_images([0, 1]),
    )


def test_partition_max_delta():
    # Set of molecules that are like: [H-H...H, H--H--H, H...H-H]
    _list = [
        Molecule(atoms=[Atom("H"), Atom("H", x=0.7), Atom("H", x=2.7)]),
        Molecule(atoms=[Atom("H"), Atom("H", x=1.35), Atom("H", x=2.7)]),
        Molecule(atoms=[Atom("H"), Atom("H", x=2.0), Atom("H", x=2.7)]),
    ]

    h2_h = NEB.from_list(_list)
    max_delta = Distance(0.1, units="Å")

    assert (
        np.max(
            np.linalg.norm(_list[0].coordinates - _list[1].coordinates, axis=1)
        )
        > max_delta
    )

    h2_h.partition(max_delta=max_delta)

    for i, j in [(0, 1), (1, 2)]:
        assert (
            np.max(
                np.linalg.norm(
                    h2_h.images[i].coordinates - h2_h.images[j].coordinates,
                    axis=1,
                )
            )
            <= max_delta
        )


def _h_xyz_string_with_energy(energy: float):
    return f"1\nE = {energy:.6f}\nH 0.0 0.0 0.0"


def _h_xyz_string():
    return f"1\ntitle line\nH 0.0 0.0 0.0"


@work_in_tmp_dir()
def test_init_from_file_sets_force_constant():
    with open("tmp.xyz", "w") as file:
        print(
            _h_xyz_string_with_energy(0.1),
            _h_xyz_string_with_energy(0.1015),
            _h_xyz_string_with_energy(0.105),
            sep="\n",
            file=file,
        )

    # Should be able to set the initial force constant
    neb = NEB.from_file("tmp.xyz", init_k=0.234)
    assert len(neb.images) == 3
    assert np.isclose(neb.init_k, 0.234)

    neb = NEB.from_file("tmp.xyz")
    # Estimated value should be reasonable
    k_1kcal_diffs = neb.init_k
    assert 0.001 < neb.init_k < 0.2

    with open("tmp.xyz", "w") as file:
        print(
            _h_xyz_string_with_energy(0.1),
            _h_xyz_string_with_energy(0.25),
            _h_xyz_string_with_energy(0.4),
            sep="\n",
            file=file,
        )

    # with larger energy differences we should have a larger k
    assert NEB.from_file("tmp.xyz").init_k > k_1kcal_diffs


@work_in_tmp_dir()
def test_init_from_file_sets_force_constant_no_energies():
    with open("tmp.xyz", "w") as file:
        print(_h_xyz_string(), _h_xyz_string(), sep="\n", file=file)

    neb = NEB.from_file("tmp.xyz")
    # Estimated value should be reasonable even without energies
    assert 0.001 < neb.init_k < 0.2


def test_neb_constructor_with_kwargs_raises():
    with pytest.raises(Exception):
        _ = NEB(init_k=ForceConstant(0.1), another_arg="a string")


def test_constructing_neb_from_endpoints_with_different_atoms_raises():
    with pytest.raises(Exception):
        _ = NEB.from_end_points(
            Molecule(smiles="O"), Molecule(smiles="C"), num=4
        )


def test_neb_from_endpoints_requires_at_least_2_images():
    # with 2 images we should be able to construct
    _ = NEB.from_end_points(
        Molecule(smiles=r"C\C=C\C"), Molecule(smiles=r"C\C=C\C"), num=2
    )

    with pytest.raises(Exception):
        _ = NEB.from_end_points(
            Molecule(smiles=r"C\C=C\C"), Molecule(smiles=r"C\C=C/C"), num=1
        )


@testutils.requires_working_xtb_install
@work_in_tmp_dir()
def test_neb_ts_guess_is_none_if_no_peak():
    init = Molecule(smiles="C")
    final = init.copy()
    final.rotate(axis=[0.1, 0.2, 0.3], theta=0.4)

    result = get_ts_guess_neb(init, final, method=XTB(), n=3)
    assert result is None
