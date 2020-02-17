from autode.methods import get_hmethod
from autode.calculation import Calculation
from autode.geom import xyz2coord, coords2xyzs
from autode.config import Config

import simtk.openmm.app as omapp
import simtk.openmm.openmm as om
from simtk.unit.unit_definitions import kelvin, angstrom, bohr, hartree, picosecond, nanometer, kilojoule_per_mole, dalton
import numpy as np
from copy import deepcopy

import math

from autode.wrappers.ORCA import ORCA


class QMMM:

    def set_up_main_simulation(self):
        # topology = omapp.Topology()
        # chain = topology.addChain()
        # residue = topology.addResidue('residue', chain)
        # atoms = []
        # for i, xyz in enumerate(self.xyzs):
        #     element = omapp.Element.getBySymbol(xyz[0])
        #     atoms.append(topology.addAtom(str(i), element, residue))
        # for atom_i, atom_j in self.bonds:
        #     topology.addBond(atoms[atom_i], atoms[atom_j])
        # self.topology = topology
        pdb = omapp.PDBFile(self.pdb_filename)
        # system = self.forcefield.createSystem(self.topology)
        system = self.forcefield.createSystem(pdb.topology)

        for xyz in self.solute_xyzs:
            el = omapp.Element.getBySymbol(xyz[0])
            mass = el.mass / dalton
            system.addParticle(mass)

        qmmm_force_obj = om.CustomExternalForce("-x*fx-y*fy-z*fz")
        qmmm_force_obj.addPerParticleParameter('fx')
        qmmm_force_obj.addPerParticleParameter('fy')
        qmmm_force_obj.addPerParticleParameter('fz')

        for i in range(self.n_atoms):
            qmmm_force_obj.addParticle(i, np.array([0.0, 0.0, 0.0]))

        system.addForce(qmmm_force_obj)

        for force in system.getForces():
            if type(force) is om.NonbondedForce:
                initial_no_exceptions = force.getNumExceptions()
                for i in range(len(self.solute_xyzs)):
                    force.addParticle(self.solute_charges[i], 0.15, 30)
                for i, xyz in enumerate(self.solute_xyzs):
                    for j, other_xyz in enumerate(self.solute_xyzs):
                        if i < j:
                            dist = np.linalg.norm(np.array(xyz[1:]) - np.array(other_xyz[1:]))
                            force.addException(i + self.n_solvent_atoms, j + self.n_solvent_atoms, 0, dist/10, 999999)
        # if self.distance_constraints is not None:
        #     for bond, distance in self.distance_constraints.items():
        #         atom_i, atom_j = bond
        #         system.addConstraint(atom_i + self.n_solvent_atoms, atom_j + self.n_solvent_atoms, distance)
        # if self.fix_qm:
        #     for i in range(len(self.qm_solvent_xyzs)):
        #         system.setParticleMass(i, 0)
        #     for i in range(len(self.solute_xyzs)):
        #         system.setParticleMass(i + self.n_solvent_atoms, 0)

        # simulation = omapp.Simulation(topology, system, self.integrator)
        simulation = omapp.Simulation(pdb.topology, system, self.integrator)

        coords_in_nm = xyz2coord(self.qm_solvent_xyzs + self.mm_solvent_xyzs + self.solute_xyzs) * 0.1

        positions = []
        for x, y, z in np.round_(coords_in_nm, 4):
            positions.append(om.Vec3(x, y, z))
        positions *= nanometer
        simulation.context.setPositions(positions)
        #simulation.reporters.append(omapp.PDBReporter('output.pdb', 1))
        simulation.minimizeEnergy()

        for force in system.getForces():
            if type(force) is om.NonbondedForce:
                exception_no = 0
                for i in range(len(self.solute_xyzs)):
                    force.setParticleParameters(i + self.n_solvent_atoms, 0, 0, 0)
                    for j in range(len(self.solute_xyzs)):
                        if i < j:
                            force.setExceptionParameters(initial_no_exceptions + exception_no, i + self.n_solvent_atoms, j + self.n_solvent_atoms, 0.000000001, 0, 0)
                            exception_no += 1
                force.updateParametersInContext(simulation.context)

        self.system = system
        self.qmmm_force_obj = qmmm_force_obj
        self.simulation = simulation

    def calc_forces_and_energies(self):
        full_mm_state = self.simulation.context.getState(getForces=True, getPositions=True, getEnergy=True)
        # print(full_mm_state.getForces())
        full_mm_energy = full_mm_state.getPotentialEnergy() * 0.00038 / kilojoule_per_mole
        positions = full_mm_state.getPositions(asNumpy=True)
        self.print_traj_point(positions)

        # qm_reg_mm_sim = self.make_sub_simulation(positions, True)
        # qm_reg_mm_state = qm_reg_mm_sim.context.getState(getForces=True, getEnergy=True)
        # qm_reg_mm_forces = qm_reg_mm_state.getForces(asNumpy=True) * bohr / hartree
        # qm_reg_mm_energy = qm_reg_mm_state.getPotentialEnergy() / hartree

        # mm_coloumb_sim = self.make_sub_simulation(positions, False)
        # mm_coloumb_state = mm_coloumb_sim.context.getState(getForces=True, getEnergy=True)
        # mm_coloumb_forces = mm_coloumb_state.getForces(asNumpy=True) * bohr / hartree
        # mm_coloumb_energy = mm_coloumb_state.getPotentialEnergy() / hartree

        qm_forces, qm_energy = self.get_qm_force_energy(positions)

        qmmm_forces = np.zeros((self.n_atoms, 3))
        for i, force in enumerate(qm_forces):
            if i < len(self.solute_xyzs):
                index = self.n_solvent_atoms + i
            else:
                index = i - len(self.solute_xyzs)
            qmmm_forces[index] += force  # - qm_reg_mm_forces[i]
        # for i, force in mm_coloumb_forces:
        #     qmmm_forces[i + self.n_qm_atoms] += force
        qmmm_forces *= hartree / bohr
        qmmm_energy = full_mm_energy + qm_energy  # - qm_reg_mm_energy + mm_coloumb_energy + qm_energy
        # print(qmmm_forces)

        self.all_qmmm_energy.append(qmmm_energy)

        return qmmm_forces

    # def make_sub_simulation(self, positions, qm_region):
    #     if qm_region:
    #         atoms = [i for i in range(self.n_qm_atoms)]
    #     else:
    #         atoms = [i for i in range(self.n_qm_atoms, self.n_atoms)]
    #     sub_topology = omapp.Topology()
    #     chain = sub_topology.addChain()
    #     residue = sub_topology.addResidue('residue', chain)
    #     sub_atoms = []
    #     for atom_no in atoms:
    #         element = omapp.Element.getBySymbol(self.xyzs[atom_no][0])
    #         sub_atoms.append(sub_topology.addAtom(str(atom_no), element, residue))
    #     for atom_i, atom_j in self.bonds:
    #         if atom_i in atoms:
    #             if qm_region:
    #                 sub_topology.addBond(sub_atoms[atom_i], atoms[atom_j])
    #             else:
    #                 atom_i_index = atom_i - self.n_qm_atoms
    #                 atom_j_index = atom_j - self.n_qm_atoms
    #                 sub_topology.addBond(sub_atoms[atom_i_index], atoms[atom_j_index])

    #     sub_system = self.forcefield.createSystem(sub_topology)
    #     if qm_region:
    #         # only include non-bonded potential
    #         for force in sub_system.getForces():
    #             if type(force) is om.NonbondedForce:
    #                 for i in range(force.getNumParticles()):
    #                     particle_params = force.getParticleParameters(i)
    #                     force.setParticleParameters(i, charge=0.0, sigma=particle_params[1], epsilon=particle_params[2])

    #     else:
    #         # only include coulomb force
    #         for i in range(sub_system.getNumForces()):
    #             if type(sub_system.getForce(i)) is not om.NonbondedForce:
    #                 sub_system.removeForce(i)
    #             else:
    #                 force = sub_system.getForce(i)
    #                 for i in range(force.getNumParticles()):
    #                     particle_params = force.getParticleParameters(i)
    #                     force.setParticleParameters(i, charge=particle_params[0], sigma=0.0, epsilon=0.0)

    #     sub_simulation = omapp.Simulation(sub_topology, sub_system, self.integrator)
    #     sub_simulation.context.setPositions(positions)

    #     return sub_simulation

    def get_qm_force_energy(self, positions):
        xyzs = self.positions2xyzs(positions)
        mol = deepcopy(self.molecule)
        mol.name = f'{self.name}_step_{self.step_no}_grad'
        mol.xyzs = xyzs[:self.n_qm_atoms]
        charges_with_coords = []
        for i in range(len(self.mm_solvent_xyzs)):
            charges_with_coords.append(xyzs[i + self.n_qm_atoms] + [self.mm_charges[i]])
        if self.hlevel:
            keywords = self.method.sp_grad_keywords
        else:
            keywords = self.method.gradients_keywords
        grad_calc = Calculation(mol.name, mol, self.method, keywords, Config.n_cores,
                                Config.max_core, charges=charges_with_coords, grad=True)
        # keywords = ORCA.gradients_keywords
        # grad_calc = Calculation(mol.name, mol, ORCA, keywords, Config.n_cores,
        #                         Config.max_core, charges=charges_with_coords, grad=True)
        grad_calc.run()
        qm_grads = grad_calc.get_gradients()  # in Eh/bohr
        qm_forces = []
        for grad in qm_grads:
            qm_forces.append([-i for i in grad])

        qm_energy = grad_calc.get_energy()

        return qm_forces, qm_energy

    def update_forces(self, forces):
        for i, force in enumerate(forces):
            self.qmmm_force_obj.setParticleParameters(i, i, force)
        self.qmmm_force_obj.updateParametersInContext(self.simulation.context)

    def take_step(self):
        self.simulation.step(1)
        self.step_no += 1

    def run_qmmm_step(self):
        self.take_step()
        state = self.simulation.context.getState(getPositions=True)
        pos = state.getPositions()
        xyzs = self.positions2xyzs(pos)
        qmmm_force = self.calc_forces_and_energies()
        self.update_forces(qmmm_force)

    def simulate(self):
        qmmm_force = self.calc_forces_and_energies()
        self.update_forces(qmmm_force)
        self.run_qmmm_step()
        while abs(self.all_qmmm_energy[-1] - self.all_qmmm_energy[-2]) > 0.00001:
            self.run_qmmm_step()
        print(self.all_qmmm_energy)
        final_state = self.simulation.context.getState(getPositions=True)
        final_positions = final_state.getPositions(asNumpy=True)
        final_xyzs = self.positions2xyzs(final_positions)

        self.final_xyzs = final_xyzs
        self.final_energy = self.all_qmmm_energy[-1]
        exit()

    def positions2xyzs(self, positions):
        coords = positions / angstrom
        coords_in_order = np.concatenate((coords[len(self.qm_solvent_xyzs) + len(self.mm_solvent_xyzs):],
                                          coords[:len(self.qm_solvent_xyzs) + len(self.mm_solvent_xyzs)]))
        return coords2xyzs(coords_in_order, self.all_xyzs)

    def print_traj_point(self, positions):
        xyzs = self.positions2xyzs(positions)
        with open(self.name + '_traj.xyz', 'a') as traj_file:
            print(f'{self.n_atoms}\n', file=traj_file)
            [print('{:<3} {:^10.5f} {:^10.5f} {:^10.5f}'.format(*line), file=traj_file) for line in xyzs]
        with open(self.name + '_qm_traj.xyz', 'a') as traj_file:
            print(f'{self.n_qm_atoms}\n', file=traj_file)
            [print('{:<3} {:^10.5f} {:^10.5f} {:^10.5f}'.format(*line), file=traj_file) for line in xyzs[:self.n_qm_atoms]]

    def __init__(self, solute, qm_solvent_xyzs, qm_solvent_bonds, mm_solvent_xyzs, mm_solvent_bonds, mm_charges, method, distance_constraints, hlevel, fix_qm):
        self.name = solute.name
        self.solute_xyzs = solute.xyzs
        self.solute_charge = solute.charge
        self.solute_mult = solute.mult
        self.solute_charges = solute.charges
        self.qm_solvent_xyzs = qm_solvent_xyzs
        self.qm_solvent_bonds = qm_solvent_bonds
        self.mm_solvent_xyzs = mm_solvent_xyzs
        self.mm_solvent_bonds = mm_solvent_bonds
        self.mm_charges = mm_charges
        self.distance_constraints = distance_constraints
        self.hlevel = hlevel
        self.fix_qm = fix_qm

        self.n_solvent_atoms = len(self.qm_solvent_xyzs) + len(self.mm_solvent_xyzs)
        self.n_qm_atoms = len(self.solute_xyzs) + len(self.qm_solvent_xyzs)
        self.all_xyzs = self.solute_xyzs + self.qm_solvent_xyzs + self.mm_solvent_xyzs
        self.n_atoms = len(self.all_xyzs)

        self.pdb_filename = self.name + '.pdb'
        xyzs2pdb(deepcopy(self.qm_solvent_xyzs + self.mm_solvent_xyzs), self.pdb_filename)

        self.molecule = deepcopy(solute)

        self.topology = None
        self.system = None
        self.simulation = None
        self.forcefield = omapp.ForceField('tip3pfb.xml')
        self.integrator = om.LangevinIntegrator(300*kelvin, 1/picosecond, 0.0005*picosecond)
        self.qmmm_force_obj = None
        self.set_up_main_simulation()

        self.method = method

        self.all_qmmm_energy = []
        self.step_no = 0

        self.final_xyzs = None
        self.final_energy = None


def xyzs2pdb(xyzs, filename):
    for i in range(len(xyzs)):
        if i % 3 == 1:
            xyzs[i][0] = 'H1'
        if i % 3 == 2:
            xyzs[i][0] = 'H2'
    with open(filename, 'w') as pdb_file:
        print('COMPND    ', 'Title\n',
              'AUTHOR    ', 'Generated by connect_waters.py', file=pdb_file)
        for i, xyz in enumerate(xyzs):
            atom_label, x, y, z = xyz
            atm = 'HETATM'
            resname = 'HOH' if atom_label in ['O', 'H1', 'H2'] else 'UNL'
            print(f'{atm:6s}{i+1:5d}{atom_label.upper():>3s}   {resname:3s}  {i//3:4d}    '
                  f'{x:8.3f}{y:8.3f}{z:8.3f}{1.0:6.2f}{0.0:6.2f}          {atom_label[0]:>2s}  ', file=pdb_file)
        # Connect the oxygen atoms
        n_o = 0
        for i, xyz in enumerate(xyzs):
            if xyz[0] == 'O':
                print(f'CONECT{i+1:5d}{i+2:5d}{i+3:5d}', file=pdb_file)
                print(f'CONECT{i+2:5d}{i+1:5d}', file=pdb_file)
                print(f'CONECT{i+3:5d}{i+1:5d}', file=pdb_file)
                n_o += 1
        print('END', file=pdb_file)
    return None
