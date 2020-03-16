from copy import deepcopy
import numpy as np
from scipy.spatial import distance_matrix
from autode import mol_graphs
from autode.species import Species
from autode.utils import requires_atoms
from autode.log import logger


class Complex(Species):

    def get_atom_indexes(self, mol_index):
        """Get the first and last atom indexes of a molecule in a Complex"""
        assert mol_index < len(self.molecules)

        first_index = sum([mol.n_atoms for mol in self.molecules[:mol_index]])
        last_index = sum([mol.n_atoms for mol in self.molecules[:mol_index + 1]])

        return list(range(first_index, last_index))

    @requires_atoms()
    def calc_repulsion(self, mol_indexes):
        """Calculate the repulsion between a molecule and the rest of the complex"""

        coordinates = self.get_coordinates()

        mol_coords = [coordinates[i] for i in mol_indexes]
        other_coords = [coordinates[i] for i in range(self.n_atoms) if i not in mol_indexes]

        # Repulsion is the sum over all pairs 1/r^4
        distance_mat = distance_matrix(mol_coords, other_coords)
        repulsion = 0.5 * np.sum(np.power(distance_mat, -4))

        return repulsion

    def __init__(self, *args):
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

        super().__init__(name='complex', atoms=complex_atoms, charge=complex_charge, mult=complex_mult)

        self.solvent = self.molecules[0].solvent
        self.graph = mol_graphs.union(graphs=[mol.graph for mol in self.molecules])


class ReactantComplex(Complex):
    pass


class ProductComplex(Complex):
    pass
