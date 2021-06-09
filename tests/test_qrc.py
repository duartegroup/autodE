import os
import numpy as np
from autode.calculation import Calculation
from autode.species import Reactant, Molecule
from autode.methods import ORCA
from autode.transition_states.base import get_displaced_atoms_along_mode
from . import testutils

here = os.path.dirname(os.path.abspath(__file__))


@testutils.work_in_zipped_dir(os.path.join(here, 'data', 'qrc.zip'))
def test_hshift_displacement():

    orca = ORCA()

    reac = Reactant(name='reactant', smiles='CC[C]([H])[H]')
    calc = Calculation(name='tmp',
                       molecule=reac,
                       method=orca,
                       keywords=orca.keywords.opt_ts)
    calc.output.filename = 'TS_hshift.out'

    assert calc.terminated_normally()
    ts = Molecule(atoms=calc.get_final_atoms())

    f_disp_atoms = get_displaced_atoms_along_mode(calc,
                                                  mode_number=6,     # TS mode
                                                  disp_factor=1.0)
    f_disp_mol = Molecule(atoms=f_disp_atoms)

    # Maximum displacement of an atom is large for this H=atom shift
    assert np.max(np.linalg.norm(ts.coordinates
                                 - f_disp_mol.coordinates, axis=1)) > 0.5

    # applying such a large shift may lead to unphysical geometries, thus
    # use a maximum scale factor
    f_disp_atoms = get_displaced_atoms_along_mode(calc,
                                                  mode_number=6,  # TS mode
                                                  disp_factor=1.0,
                                                  max_atom_disp=0.1)
    f_disp_mol = Molecule(atoms=f_disp_atoms)
    max_disp = np.max(np.linalg.norm(ts.coordinates
                                     - f_disp_mol.coordinates, axis=1))
    assert 0.05 < max_disp < 0.15
