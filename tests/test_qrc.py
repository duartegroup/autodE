import os
import numpy as np
from autode.calculations import Calculation
from autode.species import Reactant, Molecule
from autode.methods import ORCA
from autode.transition_states.base import displaced_species_along_mode
from . import testutils

here = os.path.dirname(os.path.abspath(__file__))


@testutils.work_in_zipped_dir(os.path.join(here, "data", "qrc.zip"))
def test_hshift_displacement():

    orca = ORCA()

    reac = Reactant(name="reactant", smiles="CC[C]([H])[H]")
    calc = Calculation(
        name="tmp", molecule=reac, method=orca, keywords=orca.keywords.opt_ts
    )
    calc.set_output_filename("TS_hshift.out")

    assert calc.terminated_normally
    ts = Molecule(atoms=calc.get_final_atoms())
    ts.hessian = calc.get_hessian()

    f_disp_mol = displaced_species_along_mode(
        ts, mode_number=6, disp_factor=1.0  # TS mode
    )

    # Maximum displacement of an atom is large for this H=atom shift
    assert (
        np.max(np.linalg.norm(ts.coordinates - f_disp_mol.coordinates, axis=1))
        > 0.5
    )

    # applying such a large shift may lead to unphysical geometries, thus
    # use a maximum scale factor
    f_disp_mol = displaced_species_along_mode(
        ts, mode_number=6, disp_factor=1.0, max_atom_disp=0.1  # TS mode
    )

    max_disp = np.max(
        np.linalg.norm(ts.coordinates - f_disp_mol.coordinates, axis=1)
    )
    assert 0.05 < max_disp < 0.15
