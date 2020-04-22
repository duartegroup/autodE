from copy import deepcopy
import numpy as np
from scipy.spatial import distance_matrix
from autode.solvent.explicit_solvent import do_explicit_solvent_qmmm
from autode.log import logger
from autode.mol_graphs import make_graph
from autode.mol_graphs import union
from autode.species import Species
from autode.utils import requires_atoms


class Complex(Species):

    def get_atom_indexes(self, mol_index):
        """Get the first and last atom indexes of a molecule in a Complex"""
        assert mol_index < len(self.molecules)

        first_index = sum([mol.n_atoms for mol in self.molecules[:mol_index]])
        last_index = sum([mol.n_atoms for mol in self.molecules[:mol_index + 1]])

        return list(range(first_index, last_index))

    @requires_atoms()
    def translate_mol(self, vec, mol_index):
        """
        Translate a molecule within a complex by a vector

        Arguments:
            vec (np.ndarray): Length 3 vector
            mol_index (int): Index of the molecule to translate. e.g. 2 will translate molecule 1 in the complex
                             they are indexed from 0

        """
        logger.info(f'Translating molecule {mol_index} by {vec} in {self.name}')

        for atom_index in self.get_atom_indexes(mol_index):
            self.atoms[atom_index].translate(vec)

        return None

    @requires_atoms()
    def rotate_mol(self, axis, theta, mol_index, origin=np.zeros(3)):
        """
        Rotate a molecule within a complex an angle theta about an axis given an origin

        Arguments:
            axis (np.ndarray): Length 3 vector
            theta (float): Length 3 vector
            origin (np.ndarray): Length 3 vector
            mol_index (int): Index of the molecule to translate. e.g. 2 will translate molecule 1 in the complex
                             they are indexed from 0

        """
        logger.info(f'Rotating molecule {mol_index} by {theta:.4f} radians in {self.name}')

        for atom_index in self.get_atom_indexes(mol_index):
            self.atoms[atom_index].translate(vec=-origin)
            self.atoms[atom_index].rotate(axis, theta)
            self.atoms[atom_index].translate(vec=origin)

        return None

    @requires_atoms()
    def calc_repulsion(self, mol_index):
        """Calculate the repulsion between a molecule and the rest of the complex"""

        coordinates = self.get_coordinates()

        mol_indexes = self.get_atom_indexes(mol_index)
        mol_coords = [coordinates[i] for i in mol_indexes]
        other_coords = [coordinates[i] for i in range(self.n_atoms) if i not in mol_indexes]

        # Repulsion is the sum over all pairs 1/r^4
        distance_mat = distance_matrix(mol_coords, other_coords)
        repulsion = 0.5 * np.sum(np.power(distance_mat, -4))

        return repulsion

    def __init__(self, *args, name='complex'):
        """
        Molecular complex e.g. VdW complex of one or more Molecules
        """
        self.molecules = args
        self.molecule_atom_indexes = []

        # Calculate the overall charge and spin multiplicity on the system and initialise
        complex_charge = sum([mol.charge for mol in self.molecules])
        complex_mult = sum([mol.mult for mol in self.molecules]) - (len(self.molecules) - 1)

        complex_atoms = []
        for mol in self.molecules:
            complex_atoms += deepcopy(mol.atoms)

        super().__init__(name=name, atoms=complex_atoms, charge=complex_charge, mult=complex_mult)

        self.solvent = self.molecules[0].solvent
        self.graph = union(graphs=[mol.graph for mol in self.molecules])


class ReactantComplex(Complex):

    def run_const_opt(self, const_opt, method=None, n_cores=None):
        """Run a constrained optimisation of the ReactantComplex"""
        const_opt.run()

        atoms = const_opt.get_final_atoms()
        energy = const_opt.get_energy()

        # Set the energy, new set of atoms then make the molecular graph
        self.energy = energy
        self.set_atoms(atoms=atoms)
        make_graph(species=self)

        return None


class ProductComplex(Complex):
    pass


class SolvatedReactantComplex(Complex):

    def run_const_opt(self, const_opt, method, n_cores):
        """Run a constrained optimisation of the ReactantComplex"""
        self.qm_solvent_atoms = None
        self.mm_solvent_atoms = None
        const_opt.run()

        atoms = const_opt.get_final_atoms()

        # Set the energy, new set of atoms then make the molecular graph
        self.set_atoms(atoms=atoms)

        for i, charge in enumerate(const_opt.get_atomic_charges()):
            self.graph.nodes[i]['charge'] = charge

        energy, species_atoms, qm_solvent_atoms, mm_solvent_atoms = do_explicit_solvent_qmmm(self, method, n_confs=96, n_cores=n_cores)
        self.energy = energy
        self.set_atoms(species_atoms)
        make_graph(species=self)
        self.qm_solvent_atoms = qm_solvent_atoms
        self.mm_solvent_atoms = mm_solvent_atoms

        return None

    def __init__(self, solvent_mol, *args, name='complex'):
        super().__init__(*args, name=name)
        self.solvent_mol = solvent_mol
        self.qm_solvent_atoms = None
        self.mm_solvent_atoms = None


def get_complexes(reaction):
    """Creates Reactant and Product complexes for the reaction. If it is a SolvatedReaction, a SolvatedReactantComplex is returned"""
    if reaction.__class__.__name__ == 'SolvatedReaction':
        reac = SolvatedReactantComplex(reaction.solvent_mol, *reaction.reacs, name='r')
    else:
        reac = ReactantComplex(*reaction.reacs, name='r')
    prod = ProductComplex(*reaction.prods, name='p')
    return reac, prod
