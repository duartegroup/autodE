from autode.wrappers.NWChem import NWChem
from autode.calculation import Calculation
from autode.molecule import Molecule
from autode.atoms import Atom
import numpy as np
import os

here = os.path.dirname(os.path.abspath(__file__))
test_mol = Molecule(name='methane', smiles='C')
method = NWChem()
method.available = True

opt_keywords = ['driver\n gmax 0.002\n  grms 0.0005\n'
                '  xmax 0.01\n   xrms 0.007\n  eprec 0.00003\nend',
                'basis\n  *   library Def2-SVP\nend',
                'dft\n   xc xpbe96 cpbe96\nend',
                'task dft optimize']


def test_opt_calc():
    os.chdir(os.path.join(here, 'data'))

    calc = Calculation(name='opt', molecule=test_mol, method=method,
                       keywords=opt_keywords)
    calc.run()

    assert os.path.exists('opt_nwchem.nw')
    assert os.path.exists('opt_nwchem.out')

    final_atoms = calc.get_final_atoms()
    assert len(final_atoms) == 5
    assert type(final_atoms[0]) is Atom
    assert -40.4165 < calc.get_energy() < -40.4164
    assert calc.output.exists()
    assert calc.output.file_lines is not None
    assert calc.get_imaginary_freqs() == []
    assert calc.input.filename == 'opt_nwchem.nw'
    assert calc.output.filename == 'opt_nwchem.out'
    assert calc.terminated_normally()
    assert calc.optimisation_converged()
    assert calc.optimisation_nearly_converged() is False

    charges = calc.get_atomic_charges()
    assert len(charges) == 5
    assert all(-1.0 < c < 1.0 for c in charges)

    # Optimisation should result in small gradients
    gradients = calc.get_gradients()
    assert len(gradients) == 5
    assert all(-0.1 < np.linalg.norm(g) < 0.1 for g in gradients)

    os.remove('opt_nwchem.nw')

    os.chdir(here)


def test_opt_single_atom():

    h = Molecule(name='H', smiles='[H]')
    calc = Calculation(name='opt_h', molecule=h, method=method,
                       keywords=opt_keywords)
    calc.generate_input()

    # Can't do an optimisation of a hydrogen atom..
    assert os.path.exists('opt_h_nwchem.nw')
    input_lines = open('opt_h_nwchem.nw', 'r').readlines()
    assert 'opt' not in [keyword.lower() for keyword in input_lines[0].split()]

    os.remove('opt_h_nwchem.nw')


def test_opt_hf_constraints():
    os.chdir(os.path.join(here, 'data'))

    keywords = ['driver\n gmax 0.002\n  grms 0.0005\n'
                '  xmax 0.01\n   xrms 0.007\n  eprec 0.00003\nend',
                'basis\n  *   library Def2-SVP\nend',
                'task scf optimize']

    h2o = Molecule(name='water', smiles='O')
    calc = Calculation(name='opt_water', molecule=h2o, method=method,
                       keywords=keywords,
                       cartesian_constraints=[0],
                       distance_constraints={(0, 1): 0.95})
    calc.run()
    h2o.set_atoms(atoms=calc.get_final_atoms())
    assert 0.94 < h2o.get_distance(0, 1) < 0.96

    os.remove('opt_water_nwchem.nw')

    os.chdir(here)

