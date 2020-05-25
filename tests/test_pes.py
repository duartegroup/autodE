from autode.pes_2d import PES2d
from autode.pes import get_closest_species
from autode.complex import ReactantComplex, ProductComplex
from autode.atoms import Atom
from autode.species import Species
from autode.mol_graphs import make_graph
from autode.exceptions import NoClosestSpecies
from autode.pes import FormingBond, BreakingBond
from autode.reaction import Reaction
from autode.molecule import Reactant, Product
import pytest
import numpy as np

mol = Species(name='H2', charge=0, mult=1, atoms=[Atom(atomic_symbol='H', x=0.0, y=0.0, z=0.0),
                                                  Atom(atomic_symbol='H', x=0.0, y=0.0, z=1.0)])
make_graph(mol)
reactant = ReactantComplex(mol, mol)
product = ProductComplex(mol, mol)

pes = PES2d(reactant=reactant, product=reactant,
            r1s=np.linspace(1, 3, 3), r1_idxs=(0, 1),
            r2s=np.linspace(1, 0, 2), r2_idxs=(2, 3))


def test_2d_pes_class():

    assert reactant.n_atoms == 4

    assert pes.rs.shape == (3, 2)
    assert pes.species.shape == (3, 2)

    assert pes.rs[0, 0] == (1, 1)
    assert pes.rs[1, 1] == (2, 0)
    assert pes.rs[2, 1] == (3, 0)

    assert type(pes.species[0, 0]) is ReactantComplex

    pes.species[0, 0].energy = 1

    assert pes.species[0, 0].energy == 1
    assert pes.species[1, 1] is None


def test_get_closest_species():

    assert type(get_closest_species(pes=pes, point=(0, 0))) is ReactantComplex
    assert type(get_closest_species(pes=pes, point=(1, 1))) is ReactantComplex

    pes2 = PES2d(reactant=reactant, product=product,
                 r1s=np.linspace(1, 5, 5), r1_idxs=(0, 1),
                 r2s=np.linspace(1, 5, 5), r2_idxs=(2, 3))

    with pytest.raises(NoClosestSpecies):
        get_closest_species(pes=pes2, point=(4, 4))


def test_bonds():

    h1 = Atom(atomic_symbol='H', x=0.0, y=0.0, z=0.0)
    h2 = Atom(atomic_symbol='H', x=0.0, y=0.0, z=0.7)
    h3 = Atom(atomic_symbol='H', x=0.0, y=0.0, z=1.7)

    hydrogen = Reactant(name='H2', atoms=[h1, h2], charge=0, mult=1)
    h = Reactant(name='H', atoms=[h3], charge=0, mult=2)

    reac = ReactantComplex(hydrogen, h)

    prod_h2 = Product(name='H2', atoms=[h1, h2], charge=0, mult=1)
    prod_h = Product(name='H', atoms=[h3], charge=0, mult=2)

    fbond = FormingBond(atom_indexes=(1, 2), species=reac)
    bbond = BreakingBond(atom_indexes=(0, 1), species=reac, reaction=Reaction(hydrogen, h, prod_h2, prod_h))

    assert fbond.curr_dist == 1.0
    assert 0.6 < fbond.final_dist < 0.8

    assert 2.0 < bbond.final_dist < 2.5
    assert bbond.curr_dist == 0.7
