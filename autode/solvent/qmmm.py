from copy import deepcopy
import numpy as np
from openmmtools.integrators import GradientDescentMinimizationIntegrator
import os
import simtk.openmm.app as omapp
import simtk.openmm.openmm as om
from simtk.unit.unit_definitions import kelvin, angstrom, bohr, hartree, picosecond, nanometer, kilojoule_per_mole, dalton, mole
from autode.atoms import get_vdw_radius
from autode.calculation import Calculation
from autode.config import Config
from autode.constants import Constants
from autode.log import logger
from autode.point_charges import PointCharge


class QMMM:

    def set_up_main_simulation(self, fix_solute, minimise_energy=True):
        """Set up an openmm qmmm simulation, including a qmmm force object ot use for QMMM simulations"""
        atoms2pdb(deepcopy(self.species.qm_solvent_atoms + self.species.mm_solvent_atoms), f'{self.name}.pdb')
        pdb = omapp.PDBFile(f'{self.name}.pdb')
        # TODO make the forcefield work for not water
        self.system = omapp.ForceField('tip3pfb.xml').createSystem(pdb.topology)

        coords = np.array([atom.coord for atom in self.species.qm_solvent_atoms + self.species.mm_solvent_atoms + self.species.atoms])
        box_size = (np.max(coords, axis=0) - np.min(coords, axis=0)) / 10
        x_vec = om.Vec3(box_size[0], 0, 0)
        y_vec = om.Vec3(0, box_size[1], 0)
        z_vec = om.Vec3(0, 0, box_size[2])
        self.system.setDefaultPeriodicBoxVectors(x_vec, y_vec, z_vec)

        for _ in range(self.species.n_atoms):
            self.system.addParticle(0)

        self.qmmm_force_obj = om.CustomExternalForce("-x*fx-y*fy-z*fz")
        self.qmmm_force_obj.addPerParticleParameter('fx')
        self.qmmm_force_obj.addPerParticleParameter('fy')
        self.qmmm_force_obj.addPerParticleParameter('fz')

        for i in range(len(coords)):
            self.qmmm_force_obj.addParticle(i, np.array([0.0, 0.0, 0.0]))

        self.system.addForce(self.qmmm_force_obj)

        for force in self.system.getForces():
            if type(force) is om.NonbondedForce:
                for i, atom in enumerate(self.species.atoms):
                    force.addParticle(self.species.graph.nodes[i]['charge'], 0.2 * get_vdw_radius(atom.label), 0.2)

        self.simulation = omapp.Simulation(pdb.topology, self.system, GradientDescentMinimizationIntegrator(initial_step_size=1*angstrom))

        # Prevent openmm multithreading combined with python multithreading overloading the CPU
        self.simulation.context.getPlatform().setPropertyDefaultValue('Threads', '1')
        self.simulation.context.reinitialize(preserveState=True)

        coords_in_nm = coords * 0.1
        self.simulation.context.setPositions(coords_in_nm)

        if minimise_energy:
            logger.info('Minimising solvent energy')
            self.simulation.minimizeEnergy()

        self.set_qm_atoms()

        for force in self.system.getForces():
            if type(force) is om.NonbondedForce:
                for i in range(self.species.n_atoms):
                    force.setParticleParameters(i + self.n_solvent_atoms, self.species.graph.nodes[i]['charge'], 0, 0)

        if not fix_solute:
            for i, atom in enumerate(self.species.atoms):
                self.system.setParticleMass(i + self.n_solvent_atoms, omapp.Element.getBySymbol(atom.label).mass / dalton)

            for (atom1, atom2), distance in self.dist_consts.items():
                self.system.addConstraint(atom1 + self.n_solvent_atoms, atom2 + self.n_solvent_atoms, distance/10)

    def set_qm_atoms(self):
        """Set the closest solvent molecules to be used as QM solvent"""
        positions = self.simulation.context.getState(getPositions=True).getPositions(asNumpy=True)
        solute_coords = positions[self.n_solvent_atoms:] / nanometer
        all_distances = []
        for i in range(int(self.n_solvent_atoms / self.species.solvent_mol.n_atoms)):
            solvent_coords = positions[i*self.species.solvent_mol.n_atoms:(i+1)*self.species.solvent_mol.n_atoms] / nanometer
            solvent_centre = np.average(solvent_coords, axis=0)
            distances = []
            for coord in solute_coords:
                distances.append(np.linalg.norm(coord - solvent_centre))
            all_distances.append(min(distances))
        sorted_distances = sorted(all_distances)
        closest_solvents = []
        for i in range(int(len(self.species.qm_solvent_atoms)/self.species.solvent_mol.n_atoms)):
            closest_solvents.append(all_distances.index(sorted_distances[i]))
        qm_atoms = []
        for index in closest_solvents:
            all_mol_indices = [i+(index*self.species.solvent_mol.n_atoms) for i in range(self.species.solvent_mol.n_atoms)]
            qm_atoms += all_mol_indices
        self.qm_solvent_atom_idxs = qm_atoms
        self.update_atoms(positions)

    def update_atoms(self, positions):
        """Update the species qm and mm solvent atoms lists"""
        coords = positions / angstrom
        self.species.set_coordinates(coords[self.n_solvent_atoms:])
        qm_solvent_coords = coords[self.qm_solvent_atom_idxs]
        mm_solvent_coords = coords[[i for i in range(self.n_solvent_atoms) if not i in self.qm_solvent_atom_idxs]]
        for i, coord in enumerate(qm_solvent_coords):
            self.species.qm_solvent_atoms[i].coord = coord
        for i, coord in enumerate(mm_solvent_coords):
            self.species.mm_solvent_atoms[i].coord = coord

    def simulate(self):
        """Run the QMMM simulation"""
        logger.info('Running QMMM steps')
        self.run_qmmm_step()
        self.run_qmmm_step()
        while abs(self.all_qmmm_energy[-1] - self.all_qmmm_energy[-2]) > 0.000001:
            self.run_qmmm_step()

        self.species.energy = self.all_qmmm_energy[-1]

        return None

    def run_qmmm_step(self):
        """Run a single QMMM step"""
        self.calc_forces_and_energies()
        self.simulation.step(1)
        self.set_qm_atoms()

    def calc_forces_and_energies(self):
        """Calculate the QMMM forces"""
        forces = []
        for i, force in enumerate(self.system.getForces()):
            forces.append(deepcopy(force))
            if type(force) is om.CustomExternalForce:
                qm_force_index = i

        self.system.removeForce(qm_force_index)
        self.simulation.context.reinitialize(preserveState=True)
        full_mm_state = self.simulation.context.getState(getPositions=True, getEnergy=True)
        full_mm_energy = full_mm_state.getPotentialEnergy() / (kilojoule_per_mole * Constants.ha2kJmol)
        positions = full_mm_state.getPositions(asNumpy=True)

        for _ in range(self.system.getNumForces()):
            self.system.removeForce(0)

        for force in forces:
            if type(force) is om.NonbondedForce:
                new_force = deepcopy(force)
                for i in range(new_force.getNumParticles()):
                    params = new_force.getParticleParameters(i)
                    new_force.setParticleParameters(i, params[0], 0, 0)
                self.system.addForce(new_force)
        self.simulation.context.reinitialize(preserveState=True)
        all_coulomb_state = self.simulation.context.getState(getForces=True, getEnergy=True)
        all_coulomb_energy = all_coulomb_state.getPotentialEnergy() / (kilojoule_per_mole * Constants.ha2kJmol)
        all_coulomb_forces = all_coulomb_state.getForces(asNumpy=True) * nanometer / (kilojoule_per_mole * Constants.ha2kJmol * 10)
        all_coulomb_forces = all_coulomb_forces.astype(np.float64)

        for _ in range(self.system.getNumForces()):
            self.system.removeForce(0)

        for force in forces:
            if type(force) is om.NonbondedForce:
                new_force = deepcopy(force)
                for i in range(new_force.getNumParticles()):
                    params = new_force.getParticleParameters(i)
                    if i < self.n_solvent_atoms and not i in self.qm_solvent_atom_idxs:
                        new_force.setParticleParameters(i, params[0], 0, 0)
                    else:
                        new_force.setParticleParameters(i, 0, 0, 0)
                self.system.addForce(new_force)
        self.simulation.context.reinitialize(preserveState=True)
        mm_coulomb_state = self.simulation.context.getState(getEnergy=True)
        mm_coulomb_energy = mm_coulomb_state.getPotentialEnergy() / (kilojoule_per_mole * Constants.ha2kJmol)

        for _ in range(self.system.getNumForces()):
            self.system.removeForce(0)

        for force in forces:
            self.system.addForce(force)
            if type(force) is om.CustomExternalForce:
                self.qmmm_force_obj = force
        self.simulation.context.reinitialize(preserveState=True)

        qm_forces, qm_energy = self.get_qm_force_energy()

        qmmm_forces = np.zeros((len(positions), 3))
        for i, force in enumerate(qm_forces):
            # force for solute, then solvent atoms
            if i < self.species.n_atoms:
                index = self.n_solvent_atoms + i
            else:
                index = self.qm_solvent_atom_idxs[i - self.species.n_atoms]
            qmmm_forces[index] += force
        for i, force in enumerate(all_coulomb_forces):
            if i > self.n_solvent_atoms or i in self.qm_solvent_atom_idxs:
                qmmm_forces[i] -= force
        qmmm_forces *= (kilojoule_per_mole * Constants.ha2kJmol * 10) / nanometer
        self.update_forces(qmmm_forces)

        qmmm_energy = full_mm_energy + qm_energy - all_coulomb_energy + mm_coulomb_energy
        self.all_qmmm_energy.append(qmmm_energy)

        return None

    def get_qm_force_energy(self):
        """Calculate the QM force"""
        grad_calc = Calculation(f'{self.name}_step_{self.step_no}_grad', self.species, self.method, self.method.keywords.grad, 1, grad=True,
                                point_charges=get_species_point_charges(self.species))
        grad_calc.run()
        qm_grads = grad_calc.get_gradients()  # in Eh/bohr
        qm_forces = []
        for grad in qm_grads:
            qm_forces.append([-i for i in grad])

        qm_energy = grad_calc.get_energy()
        self.final_qm_energy = qm_energy

        self.step_no += 1

        return qm_forces, qm_energy

    def update_forces(self, forces):
        """Update the QMMM force with the QM force"""
        for i, force in enumerate(forces):
            self.qmmm_force_obj.setParticleParameters(i, i, force)
        self.qmmm_force_obj.updateParametersInContext(self.simulation.context)

    def __init__(self, species, dist_consts, method, number):
        self.name = f'{species.name}_qmmm_{number}'
        self.species = species
        self.dist_consts = {} if dist_consts is None else dist_consts
        self.method = method
        self.n_solvent_atoms = len(species.qm_solvent_atoms + species.mm_solvent_atoms)

        self.qm_solvent_atom_idxs = None
        self.system = None
        self.simulation = None
        self.qmmm_force_obj = None

        self.all_qmmm_energy = []
        self.step_no = 0


def atoms2pdb(atoms, filename):
    """Creates a PDB file for a set of atoms"""
    # TODO make this work for not water
    for i, atom in enumerate(atoms):
        if i % 3 == 1:
            atom.label = 'H1'
        if i % 3 == 2:
            atom.label = 'H2'
    with open(filename, 'w') as pdb_file:
        print('COMPND    ', 'Title\n',
              'AUTHOR    ', 'Generated by autodE', file=pdb_file)
        for i, atom in enumerate(atoms):
            x, y, z = atom.coord
            atm = 'HETATM'
            resname = 'HOH'
            print(f'{atm:6s}{i+1:5d}{atom.label.upper():>3s}   {resname:3s}  {i//3:4d}    '
                  f'{x:8.3f}{y:8.3f}{z:8.3f}{1.0:6.2f}{0.0:6.2f}          {atom.label[0]:>2s}  ', file=pdb_file)
        # Add the water bonds
        for i, atom in enumerate(atoms):
            if atom.label == 'O':
                print(f'CONECT{i+1:5d}{i+2:5d}{i+3:5d}', file=pdb_file)
                print(f'CONECT{i+2:5d}{i+1:5d}', file=pdb_file)
                print(f'CONECT{i+3:5d}{i+1:5d}', file=pdb_file)
        print('END', file=pdb_file)
    return None


def get_species_point_charges(species):
    """Gets a list of point charges for a species' mm_solvent_atoms
    List has the form: float of point charge, x, y, z coordinates"""
    if not hasattr(species, 'mm_solvent_atoms') or species.mm_solvent_atoms is None:
        return None

    point_charges = []
    for i, atom in enumerate(species.mm_solvent_atoms):
        charge = species.solvent_mol.graph.nodes[i % species.solvent_mol.n_atoms]['charge']
        point_charges.append(PointCharge(charge, coord=atom.coord))

    return point_charges
