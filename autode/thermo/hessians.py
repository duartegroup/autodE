import numpy as np
from typing import List, Tuple
from autode.utils import cached_property
from autode.constants import Constants
from autode.geom import proj
from autode.values import ValueArray, Frequency, Coordinates
from autode.units import (wavenumber,
                          ha_per_ang_sq, ha_per_a0_sq, J_per_m_sq, J_per_ang_sq)


class Hessian(ValueArray):

    implemented_units = [ha_per_ang_sq, ha_per_a0_sq, J_per_m_sq, J_per_ang_sq]

    def __repr__(self):
        return f'Hessian({np.ndarray.__str__(self)} {self.units.name})'

    def __hash__(self):
        # NOTE: Required for functools.lru_cache (< Python 3.8)
        return hash(str(self))

    def __new__(cls,
                input_array,
                units=ha_per_ang_sq,
                atoms=None):
        """
        Hessian matrix

        Arguments:
            input_array (np.ndarray | autode.values.ValueArray):
            units (autode.units.Unit | str):
            atoms (list(autode.atoms.Atom) | None):

        Returns:
            (autode.thermo.hessians.Hessian):

        Raises:
            (ValueError):
        """
        arr = super().__new__(cls, input_array, units=units)

        if atoms is not None and (3*len(atoms), 3*len(atoms)) != input_array.shape:
            raise ValueError(f'Shape mismatch. Expecting '
                             f'{input_array.shape[0]//3} atoms from the Hessian'
                             f' shape, but had {len(atoms)}')
        arr.atoms = atoms

        return arr

    @cached_property
    def n_tr(self) -> int:
        """Number of translational and rotational normal modes"""
        return 5 if self.atoms.are_linear() else 6

    @cached_property
    def n_v(self) -> int:
        """Number of vibrational normal modes"""
        return 3*len(self.atoms) - self.n_tr

    def _translation_vecs(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Orthonormal translation vectors for Hessian projection

        Returns:
            (tuple(np.ndarray)):
        """
        n_atoms = len(self.atoms)

        t1 = np.tile(np.array([1., 0., 0.]), reps=n_atoms)
        t2 = np.tile(np.array([0., 1., 0.]), reps=n_atoms)
        t3 = np.tile(np.array([0., 0., 1.]), reps=n_atoms)

        return t1, t2, t3

    def _rotation_vecs(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Rotation vectors for Hessian projection, if linear then choose a
        set of orthogonal basis vectors that include the major rotation
        axis

        Returns:
            (tuple(np.ndarray)):
        """

        e_x = np.array([1., 0., 0.])
        e_y = np.array([0., 1., 0.])
        e_z = np.array([0., 0., 1.])

        com = self.atoms.com     # Centre of mass
        t4, t5, t6 = [], [], []

        for atom in self.atoms:
            t4 += np.cross(e_x, atom.coord - com).tolist()
            t5 += np.cross(e_y, atom.coord - com).tolist()
            t6 += np.cross(e_z, atom.coord - com).tolist()

        return np.array(t4), np.array(t5), np.array(t6)

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

        t1, t2, t3 = self._translation_vecs()
        t4, t5, t6 = self._rotation_vecs()

        # Construct M^1/2, which as it's diagonal, is just the roots of the
        # diagonal elements
        masses = np.repeat([atom.mass for atom in self.atoms], repeats=3,
                           axis=np.newaxis)
        m_half = np.diag(np.sqrt(masses))

        for t_i in (t1, t2, t3, t4, t5, t6):
            t_i[:] = np.dot(m_half, np.array(t_i).T)
            t_i /= np.linalg.norm(t_i)

        # Generate a transform matrix D with the first columns as translation/
        # rotation vectors with the remainder as random orthogonal columns
        M = np.random.rand(3 * len(self.atoms), 3 * len(self.atoms)) - 0.5
        M[:, :6] = np.column_stack((t1, t2, t3, t4, t5, t6))

        return np.linalg.qr(M)[0]

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
        Hessian with the translation and rotation projected out with an
        orthonormal transformation::

            H' = T^T H T

        Returns:
            (np.ndarray):
        """

        H = np.linalg.multi_dot((self._proj_matrix.T,
                                 self._mass_weighted,
                                 self._proj_matrix))
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
        n_tr = self.n_tr             # Number of translational+rotational modes
        n_v = self.n_v               # and the number of vibrations

        _, S_bar = np.linalg.eigh(self._proj_mass_weighted[n_tr:, n_tr:])

        # Re-construct the block matrix
        S_prime = np.block([[np.zeros((n_tr, n_tr)),  np.zeros((n_tr, n_v))],
                            [np.zeros((n_v, n_tr)),         S_bar        ]])

        # then apply the back-transformation
        modes = [Coordinates(np.dot(self._proj_matrix, S_prime[:, i]))
                 for i in range(6 + n_v)]

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

        n_tr = self.n_tr             # Number of translational+rotational modes
        lambdas = np.linalg.eigvalsh(self._proj_mass_weighted[n_tr:, n_tr:])

        trans_rot_freqs = [Frequency(0.0) for _ in range(n_tr)]
        vib_freqs = self._eigenvalues_to_freqs(lambdas)

        return trans_rot_freqs + vib_freqs
