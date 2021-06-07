import numpy as np
from typing import List
from autode.utils import cached_property
from autode.constants import Constants
from autode.values import ValueArray, Frequency, Coordinates
from autode.units import (wavenumber,
                          ha_per_ang_sq, ha_per_a0_sq, J_per_m_sq, J_per_ang_sq)


class Hessian(ValueArray):

    implemented_units = [ha_per_ang_sq, ha_per_a0_sq, J_per_m_sq, J_per_ang_sq]

    def __repr__(self):
        return f'Hessian({np.ndarray.__str__(self)} {self.units.name})'

    def __new__(cls,
                input_array,
                units=ha_per_ang_sq,
                atoms=None):
        """
        Hessian matrix

        Arguments:
            input_array (np.ndarray):
            units (autode.units.Unit):
            atoms (list(autode.atoms.Atom) | None):

        Raises:
            (ValueError):
        """

        arr = np.asarray(input_array).view(cls)
        arr.units = units

        if atoms is not None and (3*len(atoms), 3*len(atoms)) != input_array.shape:
            raise ValueError(f'Shape mismatch. Expecting '
                             f'{input_array.shape[0]//3} atoms from the Hessian'
                             f' shape, but had {len(atoms)}')

        arr.atoms = atoms

        return arr

    @cached_property
    def _proj_matrix(self) -> np.ndarray:
        """
        Construct the projection matrix to transform the Hessian into block
        diagonal form::

            H  => ( 0   0)
                  (0    H')

            F = D^T H D

        Method from:
        https://chemistry.stackexchange.com/questions/74639/how-to-calculate
        -wavenumbers-of-normal-modes-from-the-eigenvalues-of-the-cartesi

        see common/hessians.tex for methods

        Returns:
            (np.ndarray): Transform matrix (D)
        """
        n_atoms = len(self.atoms)

        e_x = np.array([1., 0., 0.])
        e_y = np.array([0., 1., 0.])
        e_z = np.array([0., 0., 1.])

        d1 = np.tile(e_x, reps=n_atoms)  # Translation vectors
        d2 = np.tile(e_y, reps=n_atoms)
        d3 = np.tile(e_z, reps=n_atoms)

        com = self.atoms.com  # Centre of mass
        d4, d5, d6 = [], [], []  # Rotation vectors

        for atom in self.atoms:
            d4 += np.cross(e_x, atom.coord - com).tolist()
            d5 += np.cross(e_y, atom.coord - com).tolist()
            d6 += np.cross(e_z, atom.coord - com).tolist()

        # Construct M^1/2, which as it's diagonal, is just the roots of the
        # diagonal elements
        masses = np.repeat([atom.mass for atom in self.atoms], repeats=3,
                           axis=np.newaxis)
        m_half = np.diag(np.sqrt(masses))

        for col in (d1, d2, d3, d4, d5, d6):
            col[:] = np.dot(m_half, np.array(col).T)
            col /= np.linalg.norm(col)

        # Generate a transform matrix D with the first columns as translation/
        # rotation vectors with the remainder as random orthogonal columns
        D = np.random.rand(3 * n_atoms, 3 * n_atoms) - 0.5

        D[:, :6] = np.column_stack((d1, d2, d3, d4, d5, d6))
        return np.linalg.qr(D)[0]

    @cached_property
    def _mass_weighted(self) -> np.ndarray:
        """Mass weighted the Hessian matrix

                      H_ij
        H'_ij  =  ------------
                   √(m_i x m_j)
        """
        if self.atoms is None:
            raise ValueError('Could not calculate frequencies. Atoms not set')

        H = self.to('J ang^-2')
        mass_array = np.repeat([atom.mass.to('kg') for atom in self.atoms],
                               repeats=3,
                               axis=np.newaxis)

        return H / np.sqrt(np.outer(mass_array, mass_array))

    @cached_property
    def _proj_mass_weighted(self) -> np.ndarray:
        """
        Sub-Hessian with the translation and rotation projected out.

        Returns:
            (np.ndarray):
        """
        D = self._proj_matrix
        H = np.linalg.multi_dot((D.T, self._mass_weighted, D))

        return H

    @cached_property
    def normal_modes(self) -> List[Coordinates]:
        """
        Calculate the normal modes as the eigenvectors of the Hessian matrix

        Returns:
            (list(autode.values.Coordinates)):

        Raises:
            (ValueError): If atoms are not set
        """
        _, modes = np.linalg.eigh(self._mass_weighted)

        # Convert the eigenvectors from the columns of the matrix into
        # individual displacements that can be added to a set of coordinates
        return [Coordinates(mode) for mode in modes.T]

    @cached_property
    def normal_modes_proj(self) -> List[Coordinates]:
        """
        Normal modes from the projected Hessian without rotation or translation

        Returns:
            (list(autode.values.Coordinates)):
        """
        n_tr = 6                     # Number of translational+rotational modes
        n_vibs = 3*len(self.atoms) - n_tr        # and the number of vibrations

        _, L_prime = np.linalg.eigh(self._proj_mass_weighted[n_tr:, n_tr:])

        # Re-construct the block matrix
        L = np.block([[np.zeros((n_tr, n_tr)),     np.zeros((n_tr, n_vibs))],
                      [np.zeros((n_vibs, n_tr)),           L_prime         ]])

        # then apply the back-transformation
        modes = [Coordinates(np.dot(self._proj_matrix, L[:, i]))
                 for i in range(6 + n_vibs)]

        return modes

    @staticmethod
    def _eigenvalues_to_freqs(lambdas) -> List[Frequency]:
        """
        Convert eigenvalues of the Hessian matrix (SI units) to
        frequencies in wavenumber units

        Arguments:
            lambdas (np.ndarray):

        Returns:
            (list(autode.values.Frequency))
        """

        nus = (np.sqrt(np.complex_(lambdas))
               / (2.0 * np.pi * Constants.ang_to_m * Constants.c_in_cm))

        # Cast the purely complex eigenvalues to negative real numbers, as is
        # usual in quantum chemistry codes
        nus[np.iscomplex(nus)] = -np.abs(nus[np.iscomplex(nus)])

        return [Frequency(np.real(nu), units=wavenumber) for nu in nus]

    @cached_property
    def frequencies(self) -> List[Frequency]:
        """
        Calculate the normal mode frequencies from the eigenvalues of the
        Hessian matrix

        Returns:
            (list(autode.values.Frequency)):

        Raises:
            (ValueError): If atoms are not set
        """
        lambdas = np.linalg.eigvalsh(self._mass_weighted)
        freqs = self._eigenvalues_to_freqs(lambdas)

        return freqs

    @cached_property
    def frequencies_proj(self) -> List[Frequency]:
        """
        Frequencies with rotation and translation projected out

        Returns:
            (list(autode.values.Frequency))
        """
        lambdas = np.linalg.eigvalsh(self._proj_mass_weighted[6:, 6:])

        trans_rot_freqs = [Frequency(0.0) for _ in range(6)]
        vib_freqs = self._eigenvalues_to_freqs(lambdas)

        return trans_rot_freqs + vib_freqs

    @cached_property
    def frequencies_lproj(self):
        """
        Frequencies with rotation and translation projected out for linear
        species

        Returns:

        """
        raise NotImplementedError
