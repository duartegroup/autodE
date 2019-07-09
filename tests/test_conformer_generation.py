import os
from rdkit import Chem
from rdkit.Chem import AllChem
from autode import Reactant
from autode import confomers
here = os.path.dirname(os.path.abspath(__file__))


def test_confs():

    propane = Reactant(name='propane')
    propane.mol_obj = Chem.MolFromSmiles('CCC')
    propane.conf_ids = list(AllChem.EmbedMultipleConfs(propane.mol_obj, numConfs=3, params=AllChem.ETKDG()))
    conf_xyzs = confomers.gen_conformer_xyzs(propane, conf_ids=propane.conf_ids)

    assert len(conf_xyzs) == 3              # 3 conformers
    assert len(conf_xyzs[0]) == 3           # 3 atoms
    assert len(conf_xyzs[0][0]) == 4        # atom id, x, y, z
    assert type(conf_xyzs[0][0][0]) == str
    assert all([type(conf_xyzs[0][0][i]) == float for i in [1, 2, 3]])


def test_conformer_class():

    os.chdir(os.path.join(here, 'conformers'))

    conf = confomers.Conformer()

    assert conf.name == 'conf'
    assert conf.xyzs is None
    assert conf.energy is None
    assert conf.solvent is None
    assert conf.charge == 0
    assert conf.mult == 1

    conf.xyzs = [['H', 0.0, 0.0, 0.0], ['H', 1.0, 0.0, 0.0]]
    conf.xtb_optimise()
    assert conf.energy == -0.982686092499

    conf.orca_optimise()
    assert conf.energy == -1.160780546267

