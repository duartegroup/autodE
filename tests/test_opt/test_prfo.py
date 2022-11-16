import numpy as np
from autode.species.molecule import Molecule
from autode.atoms import Atom
from autode.methods import XTB
from autode.opt.optimisers import PRFOptimiser
from autode.utils import work_in_tmp_dir
from ..testutils import requires_with_working_xtb_install

xtb = XTB()


def has_single_imag_freq_at_xtb_level(mol: Molecule) -> bool:
    mol.calc_hessian(method=xtb)

    return len(mol.imaginary_frequencies) == 1


@requires_with_working_xtb_install
@work_in_tmp_dir()
def test_sn2_opt():

    mol = Molecule(
        name="sn2_ts",
        charge=-1,
        solvent_name="water",
        atoms=[
            Atom("F", -4.17085, 3.55524, 1.59944),
            Atom("Cl", -0.75962, 3.53830, -0.72354),
            Atom("C", -2.51988, 3.54681, 0.47836),
            Atom("H", -3.15836, 3.99230, -0.27495),
            Atom("H", -2.54985, 2.47411, 0.62732),
            Atom("H", -2.10961, 4.17548, 1.25945),
        ],
    )

    assert mol.is_implicitly_solvated

    PRFOptimiser.optimise(mol, method=xtb, maxiter=10, init_alpha=0.02)

    assert has_single_imag_freq_at_xtb_level(mol)
    freq = mol.imaginary_frequencies[0]
    assert np.isclose(freq.to("cm-1"), -555, atol=20)


@requires_with_working_xtb_install
@work_in_tmp_dir()
def test_diels_alder_ts_opt():

    xyz_file_string = (
        "16\n\n"
        "C   -0.00246842   1.65107949   0.05871997\n"
        "C    1.19010335   1.11169078   0.27709357\n"
        "C    1.58518503  -0.30014150   0.31048716\n"
        "C    0.05830748  -1.54292177  -0.45110322\n"
        "C   -1.18801295  -1.07571606   0.14240225\n"
        "C   -1.28206230   0.99883443  -0.23631125\n"
        "H   -0.07432316   2.73634443   0.08639466\n"
        "H    2.01755127   1.78921170   0.47735384\n"
        "H    1.70502646  -0.70915918   1.30549866\n"
        "H    2.40397525  -0.55376353  -0.34855409\n"
        "H    0.44229481  -2.48695410  -0.08638411\n"
        "H    0.15288739  -1.41865071  -1.51944246\n"
        "H   -1.25409868  -1.13318437   1.21833314\n"
        "H   -2.09996454  -1.35917816  -0.36714627\n"
        "H   -2.09461648   1.29054940   0.41494506\n"
        "H   -1.56001451   1.00182912  -1.28216692\n"
    )
    with open("init.xyz", "w") as file:
        print(xyz_file_string, file=file)

    mol = Molecule("init.xyz")
    PRFOptimiser.optimise(mol, method=xtb, maxiter=50, init_alpha=0.05)
    assert has_single_imag_freq_at_xtb_level(mol)
    # print(mol.imaginary_frequencies)  # should be ~600 cm-1
