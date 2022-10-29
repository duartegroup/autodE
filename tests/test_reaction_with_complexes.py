import os
import shutil
import autode as ade
from . import testutils

here = os.path.dirname(os.path.abspath(__file__))


@testutils.work_in_zipped_dir(
    os.path.join(here, "data", "reaction_with_complexes.zip")
)
@testutils.requires_with_working_xtb_install
def test_reaction_w_complexes():

    ade.Config.n_cores = 1  # Ensure only a single core is used

    ade.Config.hcode = "orca"
    ade.Config.ORCA.path = here  # Spoof ORCA install

    # Ensure no DFT needs to be done other than that saved
    ade.Config.num_conformers = 1
    ade.Config.max_num_complex_conformers = 1
    ade.Config.ts_template_folder_path = os.getcwd()

    f = ade.Reactant(name="f", smiles="[F-]")
    mecl = ade.Reactant(name="mecl", smiles="ClC")
    cl = ade.Product(name="cl", smiles="[Cl-]")
    mef = ade.Product(name="mef", smiles="CF")

    rxn = ade.Reaction(f, mecl, cl, mef, solvent_name="water", name="sn2_wc")
    rxn.calculate_reaction_profile(with_complexes=True)

    assert rxn.ts is not None

    # Should have a defined energy for the reactant and product complexes
    assert rxn.reactant.energy is not None
    assert rxn.reactant.n_molecules == 2

    assert rxn.product.energy is not None
    assert rxn.product.n_molecules == 2
