from autode.pes.pes_1d import get_ts_guess_1d, PES1d
from autode.atoms import Atom
from autode.species.molecule import Molecule
from autode.bonds import FormingBond
from autode.species.complex import ReactantComplex, ProductComplex
from autode.config import Config
from autode.wrappers.ORCA import orca
from autode.wrappers.keywords import OptKeywords
from . import testutils
from autode.utils import work_in
import numpy as np
import os

here = os.path.dirname(os.path.abspath(__file__))
Config.high_quality_plots = False

opt_keywords = OptKeywords(['PBE', 'def2-SVP', 'Opt'])

# Set the reactant and product complexes

h1 = Atom(atomic_symbol='H', x=0.0, y=0.0, z=0.0)
h2 = Atom(atomic_symbol='H', x=0.0, y=0.0, z=0.7)
h3 = Atom(atomic_symbol='H', x=0.0, y=0.0, z=1.7)

hydrogen = Molecule(name='H2', atoms=[h1, h2], charge=0, mult=1)
h = Molecule(name='H', atoms=[h3], charge=0, mult=2)

reac = ReactantComplex(hydrogen, h)
prod = ProductComplex(h, hydrogen)


@testutils.unzip_dir(os.path.join(here, 'data', 'pes1d.zip'))
@work_in(os.path.join(here, 'data'))
def test_get_ts_guess_1dscan():

    fbond = FormingBond(atom_indexes=(1, 2), species=reac)
    fbond.final_dist = 0.7

    ts_guess = get_ts_guess_1d(name='H+H2_H2+H',
                               reactant=reac, product=prod,
                               bond=fbond,
                               method=orca,
                               keywords=opt_keywords,
                               dr=0.06)

    assert ts_guess.n_atoms == 3

    # Peak in the PES is at r = 0.85 Å
    assert 0.84 < ts_guess.distance(1, 2) < 0.86
    assert os.path.exists('H+H2_H2+H.png')
    os.remove('H+H2_H2+H.png')


@testutils.unzip_dir(os.path.join(here, 'data', 'pes1d.zip'))
@work_in(os.path.join(here, 'data'))
def test_1d_pes():

    pes = PES1d(reactant=reac, product=prod, rs=np.linspace(1.0, 0.7, 5),
                r_idxs=(1, 2))

    assert pes.n_points == 5
    assert type(pes.rs) is np.ndarray
    assert pes.rs[0] == 1.0
    assert pes.rs[-1] == 0.7

    # First point on the surface is the reactant, and others are initialised
    # as None
    assert type(pes.species[0]) is ReactantComplex
    assert all(s is None for s in pes.species[1:])

    assert len(pes.rs_idxs) == 1
    assert pes.rs_idxs[0] == (1, 2)

    assert pes.products_made()

    pes.calculate(name='H+H2_H2+H', method=orca,
                  keywords=opt_keywords)

    assert all(s is not None for s in pes.species)
    assert all(hasattr(s, 'energy') for s in pes.species)
    assert all(hasattr(s, 'atoms') for s in pes.species)

    # Make the plot and ensure the file exists
    pes.print_plot(method_name='orca', name='H+H2_H2+H')
    assert os.path.exists('H+H2_H2+H.png')
    os.remove('H+H2_H2+H.png')
