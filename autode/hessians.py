"""
Hessian diagonalisation and projection routines. See autode/common/hessians.pdf
for mathematical background
"""
import numpy as np
from typing import List, Tuple, Iterator, Optional
from multiprocessing import Pool
from autode.wrappers.keywords import GradientKeywords
from autode.log import logger
from autode.utils import cached_property
from autode.config import Config
from autode.constants import Constants
from autode.values import ValueArray, Frequency, Coordinates
from autode.utils import work_in, hashable
from autode.units import (wavenumber,
                          ha_per_ang_sq, ha_per_a0_sq, J_per_m_sq, J_per_ang_sq,
                          J_per_ang_sq_kg)


class Hessian(ValueArray):

    implemented_units = [ha_per_ang_sq, ha_per_a0_sq, J_per_m_sq, J_per_ang_sq,
                         J_per_ang_sq_kg]

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
        """
        5 for a linear molecule and 6 otherwise (3 rotation, 3 translation)

        Returns:
            (int): Number of translational and rotational normal modes

        Raises:
            (ValueError): Without atoms set
        """
        if self.atoms is None or not hasattr(self.atoms, 'are_linear'):
            raise ValueError('Could not determine the number of translations'
                             'and rotations. Atoms must be set')

        return 5 if self.atoms.are_linear() else 6

    @cached_property
    def n_v(self) -> int:
        """
        3N-6 for a non-linear molecule with N atoms

        Returns:
            (int): Number of vibrational normal modes

        Raises:
            (ValueError): Without atoms set
        """
        if self.atoms is None:
            raise ValueError('Could not determine the number of vibrations.'
                             ' Atoms must be set')

        return 3*len(self.atoms) - self.n_tr

    def _tr_vecs(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray,
                                np.ndarray, np.ndarray, np.ndarray]:
        """
        Orthonormal translation and rotation (tr) vectors for Hessian
        projection.

        Returns:
            (tuple(np.ndarray)):

        Raises:
            (RecursionError): If an orthogonal set cannot be constructed
        """
        n_atoms = len(self.atoms)

        # Get a random orthonormal basis in 3D
        (e_x, e_y, e_z), _ = np.linalg.qr(np.random.rand(3, 3))

        t1 = np.tile(e_x, reps=n_atoms)
        t2 = np.tile(e_y, reps=n_atoms)
        t3 = np.tile(e_z, reps=n_atoms)

        com = self.atoms.com               # Centre of mass
        t4, t5, t6 = [], [], []

        for atom in self.atoms:
            t4 += np.cross(e_x, atom.coord - com).tolist()
            t5 += np.cross(e_y, atom.coord - com).tolist()
            t6 += np.cross(e_z, atom.coord - com).tolist()

        if any(np.isclose(np.linalg.norm(t_i), 0.0) for t_i in (t4, t5, t6)):
            # Found linear dependency in rotation vectors, attempt to remove
            # by initialising different random orthogonal vectors
            return self._tr_vecs()

        return t1, t2, t3, np.array(t4), np.array(t5), np.array(t6)

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
        if self.atoms is None:
            raise ValueError('Could generate projected Hessian. Atoms not set')

        t1, t2, t3, t4, t5, t6 = self._tr_vecs()

        # Construct M^1/2, which as it's diagonal, is just the roots of the
        # diagonal elements
        masses = np.repeat([atom.mass for atom in self.atoms], repeats=3,
                           axis=np.newaxis)
        m_half = np.diag(np.sqrt(masses))

        for t_i in (t1, t2, t3, t4, t5, t6):

            t_i[:] = np.dot(m_half, np.array(t_i))
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

        return Hessian(H / np.sqrt(np.outer(mass_array, mass_array)),
                       units='J m^-2 kg^-1')

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
        return [Coordinates(mode / np.linalg.norm(mode)) for mode in modes.T]

    @cached_property
    def normal_modes_proj(self) -> List[Coordinates]:
        """
        Normal modes from the projected Hessian without rotation or translation

        Returns:
            (list(autode.values.Coordinates)):
        """
        if self.atoms is None:
            raise ValueError('Could not calculate projected normal modes, must'
                             ' have atoms set')

        n_tr = self.n_tr             # Number of translational+rotational modes
        n_v = self.n_v               # and the number of vibrations

        _, S_bar = np.linalg.eigh(self._proj_mass_weighted[n_tr:, n_tr:])

        # Re-construct the block matrix
        S_prime = np.block([[np.zeros((n_tr, n_tr)),  np.zeros((n_tr, n_v))],
                            [np.zeros((n_v, n_tr)),         S_bar        ]])

        # then apply the back-transformation
        modes = []
        for i in range(n_tr + n_v):
            mode = np.dot(self._proj_matrix, S_prime[:, i])

            # only normalise the vibrations as the rotations/translations are 0
            if i >= n_tr:
                mode /= np.linalg.norm(mode)

            modes.append(Coordinates(mode))

        return modes

    @staticmethod
    def _eigenvalues_to_freqs(lambdas) -> List[Frequency]:
        """
        Convert eigenvalues of the Hessian matrix (SI units) to
        frequencies in wavenumber units

        Arguments:
            lambdas (np.ndarray):

        Returns:
            (list(autode.values.Frequency)):
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
            (ValueError): Without atoms set
        """
        lambdas = np.linalg.eigvalsh(self._mass_weighted)
        freqs = self._eigenvalues_to_freqs(lambdas)

        return freqs

    @cached_property
    def frequencies_proj(self) -> List[Frequency]:
        """
        Frequencies with rotation and translation projected out

        Returns:
            (list(autode.values.Frequency)):

        Raises:
            (ValueError): Without atoms set
        """
        if self.atoms is None:
            raise ValueError('Could not calculate projected frequencies, must '
                             'have atoms set')

        n_tr = self.n_tr             # Number of translational+rotational modes
        lambdas = np.linalg.eigvalsh(self._proj_mass_weighted[n_tr:, n_tr:])

        trans_rot_freqs = [Frequency(0.0) for _ in range(n_tr)]
        vib_freqs = self._eigenvalues_to_freqs(lambdas)

        return trans_rot_freqs + vib_freqs

    def copy(self) -> 'Hessian':
        return self.__class__(np.copy(self), units=self.units, atoms=self.atoms)


class NumericalHessianCalculator:

    def __init__(self,
                 species:   'autode.species.Species',
                 method:    'autode.wrappers.base.Method',
                 keywords:  'autode.wrappers.keywords.GradientKeywords',
                 do_c_diff:  bool,
                 shift:      'autode.values.Distance',
                 n_cores:    Optional[int] = None
                 ):

        self._species = species
        self._method = method
        self._keywords = self._validated(keywords)

        self._do_c_diff = do_c_diff
        self._shift = shift

        self._hessian = Hessian(np.zeros(shape=self._hessian_shape),
                                units='Ha Å^-2',
                                atoms=species.atoms.copy())

        self._calculated_rows = []

        self._n_total_cores = Config.n_cores if n_cores is None else n_cores

    @work_in('numerical_hessian')
    def calculate(self, ) -> None:
        """Calculate the Hessian"""

        logger.info(f'Calculating a numerical Hessian '
                    f'{"with" if self._do_c_diff else "without"} central '
                    f'differences using {self._n_total_cores} total cores.\n'
                    f'Doing: {self._n_rows * (2 if self._do_c_diff else 1)} '
                    f'gradient evaluations')

        if not self._do_c_diff:
            logger.info('Calculating gradient at current point')
            self._init_gradient = self._gradient(species=self._species)

        # Although n_rows may be < n_cores there will not be > n_rows processes
        with Pool(processes=self._n_total_cores) as pool:

            func_name = '_cdiff_row' if self._do_c_diff else '_diff_row'

            res = [pool.apply_async(hashable(func_name, self),
                                    args=(i, k))
                   for (i, k) in self._idxs_to_calculate()]

            for row_idx, row in enumerate(res):
                self._hessian[row_idx, :] = row.get(timeout=None)

        return None

    @property
    def hessian(self) -> Hessian:
        """Hessian matrix of {d^2E/dX_ij^2}. Must be symmetric"""

        arr = np.array(self._hessian, copy=True)
        self._hessian[:] = (arr + arr.T) / 2.0

        return self._hessian

    @property
    def _hessian_shape(self) -> Tuple[int, int]:
        """Shape of the Hessian matrix for the species"""
        return 3 * self._species.n_atoms, 3 * self._species.n_atoms

    @property
    def _n_rows(self) -> int:
        """Number of rows in the Hessian"""
        return 3 * self._species.n_atoms

    @property
    def _n_cores_pp(self) -> int:
        """Number of cores per process to use e.g. a 6x6 Hessian with
        Config.n_cores = 12 -> _n_cores = 2"""
        return max(self._n_total_cores // self._n_rows, 1)

    def _new_species(self, atom_idx: int, component: int, direction: str):
        """
        New species with an applied shift to an atom. For example, water_0x+
        for a water molecule where the 0th atom has been shifted in the
        positive x direction
        """
        assert direction in ('+', '-')    # Positive or negative

        species = self._species.new_species()

        c = ['x', 'y', 'z'][component]
        species.name = f'{self._species.name}_{atom_idx}{c}{direction}'

        vec = self._shift_vector(component=component)
        species.atoms[atom_idx].translate(vec if direction == '+' else -vec)

        return species

    def _idxs_to_calculate(self) -> Iterator:
        """Generate the indexes of atoms and cartesian components that
         need to be calculated"""

        for row_idx in range(self._n_rows):

            if row_idx not in self._calculated_rows:
                self._calculated_rows.append(row_idx)

                atom_idx = row_idx // 3
                component = row_idx % 3    # 0: x, 1: y, 2: z

                yield atom_idx, component

        return

    def _validated(self,
                   keywords: 'autode.keywords.Keywords'
                   ) -> 'autode.keywords.GradientKeywords':
        """Validate the keywords"""

        if not isinstance(keywords, GradientKeywords):
            raise ValueError('Numerical Hessian require the keywords to be '
                             'GradientKeywords')

        if keywords.contain_any_of('hess', 'freq', 'hessian', 'frequency'):
            raise ValueError('Cannot calculate numerical Hessian with keywords'
                             ' that contain Hess or Freq. Must be only grad')

        return keywords

    def _shift_vector(self, component: int) -> np.ndarray:
        """Vector to shift an atom by along a defined Cartesian component,
        where, for example, component=0 -> x translation in +h direction"""

        vec = np.zeros(shape=(3,))
        vec[component] += float(self._shift)

        return vec

    def _gradient(self, species) -> 'autode.values.Gradient':
        """Evaluate the flat gradient, with shape = (3 n_atoms,) """
        from autode.calculation import Calculation

        calc = Calculation(name=species.name,
                           molecule=species,
                           method=self._method,
                           keywords=self._keywords,
                           n_cores=self._n_cores_pp)
        calc.run()

        return calc.get_gradients().flatten()

    @property
    def _init_gradient(self) -> 'autode.values.Gradient':
        """Gradient at the initial geometry of the species"""
        return np.array(self._species.gradient).flatten()

    @_init_gradient.setter
    def _init_gradient(self, value):
        """Set the initial gradient"""
        self._species.gradient = value.reshape(self._species.n_atoms, 3)

    def _cdiff_row(self, atom_idx, component) -> np.ndarray:
        """Calculate a Hessian row with central differences"""

        s_plus = self._new_species(atom_idx, component, direction='+')
        s_minus = self._new_species(atom_idx, component, direction='-')

        row = ((self._gradient(s_plus) - self._gradient(s_minus))
               / (2 * self._shift))

        return row

    def _diff_row(self, atom_idx, component) -> np.ndarray:
        """Calculate a Hessian row with one-sided differences"""

        s_plus = self._new_species(atom_idx, component, direction='+')

        row = ((self._gradient(s_plus) - self._init_gradient)
               / self._shift)

        return row
