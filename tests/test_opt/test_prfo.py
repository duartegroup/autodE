import numpy as np
from autode.species.molecule import Molecule
from autode.atoms import Atom
from autode.methods import XTB
from autode.opt.optimisers import PRFOptimiser
from autode.utils import work_in_tmp_dir
from ..testutils import requires_with_working_xtb_install


@requires_with_working_xtb_install
@work_in_tmp_dir()
def test_sn2_opt():

    xtb = XTB()

    mol = Molecule(name='sn2_ts', charge=-1, solvent_name='water',
                   atoms=[Atom('F', -4.17085, 3.55524, 1.59944),
                          Atom('Cl', -0.75962, 3.53830, -0.72354),
                          Atom('C', -2.51988, 3.54681,  0.47836),
                          Atom('H', -3.15836, 3.99230, -0.27495),
                          Atom('H', -2.54985, 2.47411,  0.62732),
                          Atom('H', -2.10961, 4.17548,  1.25945)])

    assert mol.is_implicitly_solvated

    PRFOptimiser.optimise(mol, method=xtb, maxiter=10, init_alpha=0.02)
    mol.calc_hessian(method=xtb)

    assert len(mol.imaginary_frequencies) == 1
    freq = mol.imaginary_frequencies[0]

    assert np.isclose(freq.to('cm-1'),
                      -555,
                      atol=20)
