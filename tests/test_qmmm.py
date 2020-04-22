from autode.solvent.qmmm import QMMM
from autode.solvent.qmmm import atoms2pdb
from autode.solvent.explicit_solvent import add_solvent_molecules
from autode.molecule import SolvatedMolecule
from autode.atoms import Atom
from autode.wrappers.XTB import xtb
import os
from simtk.openmm.app import Simulation
from simtk.openmm.openmm import System
from simtk.openmm.openmm import CustomExternalForce


here = os.path.dirname(os.path.abspath(__file__))
xtb.available = True


def test_qmmm():
    os.chdir(os.path.join(here, 'data'))

    qmmm_mol = SolvatedMolecule(name='mol', atoms=[Atom('F', 0.0, 0.0, 0.0)], solvent_name='water', charge=-1)
    qmmm_mol.graph.nodes()[0]['charge'] = -1
    water = SolvatedMolecule(atoms=[Atom('O', 0.0, 0.0, 0.0), Atom('H', 0.7, 0.7, 0.0), Atom('H', -0.7, 0.7, 0.0)])
    water.graph.nodes()[0]['charge'] = -0.5
    water.graph.nodes()[1]['charge'] = 0.25
    water.graph.nodes()[2]['charge'] = 0.25
    qmmm_mol.solvent_mol = water
    add_solvent_molecules(qmmm_mol, 3, 6)

    qmmm = QMMM(qmmm_mol, None, xtb, 0)
    assert qmmm.name == 'mol_qmmm_0'
    assert qmmm.dist_consts == {}
    assert qmmm.n_solvent_atoms == 18

    qmmm.set_up_main_simulation(fix_solute=False, minimise_energy=False)
    assert qmmm.qm_solvent_atom_idxs == [0, 1, 2, 3, 4, 5, 6, 7, 8]
    assert isinstance(qmmm.system, System)
    assert isinstance(qmmm.simulation, Simulation)
    assert isinstance(qmmm.qmmm_force_obj, CustomExternalForce)

    qm_force, qm_energy = qmmm.get_qm_force_energy()
    assert qm_energy == -18.847923248787
    assert qm_force[0] == [-0.05682036, -0.50133117, -0.54210066]
    assert qmmm.step_no == 1
    assert os.path.exists('mol_qmmm_0_step_0_grad_xtb.grad')
    assert os.path.exists('mol_qmmm_0_step_0_grad_xtb.pc')

    os.remove('mol_qmmm_0_step_0_grad_xtb.pc')
    os.remove('mol_qmmm_0_step_0_grad_xtb.xyz')
    os.remove('xcontrol_mol_qmmm_0_step_0_grad')
    os.remove('mol_qmmm_0.pdb')
    os.chdir(here)


def test_atoms2pdb():

    atoms2pdb([Atom('O', 0.0, 0.0, 0.0), Atom('H', 0.7, 0.7, 0.0), Atom('H', -0.7, 0.7, 0.0)], 'test.pdb')

    assert os.path.exists('test.pdb')
    pdb_file_lines = open('test.pdb', 'r').readlines()
    assert pdb_file_lines[2].split() == ['HETATM', '1', 'O', 'HOH', '0', '0.000', '0.000', '0.000', '1.00', '0.00', 'O']
    assert pdb_file_lines[3].split() == ['HETATM', '2', 'H1', 'HOH', '0', '0.700', '0.700', '0.000', '1.00', '0.00', 'H']
    assert pdb_file_lines[5].split() == ['CONECT', '1', '2', '3']

    os.remove('test.pdb')
