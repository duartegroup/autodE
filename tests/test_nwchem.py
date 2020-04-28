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


def test_opt_calc():
    os.chdir(os.path.join(here, 'data'))

    opt_keywords = ['driver\n gmax 0.002\n  grms 0.0005\n'
                    '  xmax 0.01\n   xrms 0.007\n  eprec 0.00003\nend',
                    'basis\n  *   library Def2-SVP\nend',
                    'dft\n   xc xpbe96 cpbe96\nend',
                    'task dft optimize']

    calc = Calculation(name='opt', molecule=test_mol, method=method,
                       keywords_list=opt_keywords)
    calc.run()

    assert os.path.exists('opt_nwchem.nw') is True
    assert os.path.exists('opt_nwchem.out') is True

    final_atoms = calc.get_final_atoms()
    assert len(final_atoms) == 5
    assert type(final_atoms[0]) is Atom
    assert -40.4165 < calc.get_energy() < -40.4164
    assert calc.output_file_exists is True
    assert calc.rev_output_file_lines is not None
    assert calc.output_file_lines is not None
    assert calc.get_imag_freqs() == []
    assert calc.input_filename == 'opt_nwchem.nw'
    assert calc.output_filename == 'opt_nwchem.out'
    assert calc.terminated_normally is True
    assert calc.calculation_terminated_normally() is True
    assert calc.optimisation_converged() is True
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
