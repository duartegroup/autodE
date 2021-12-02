"""
Potential energy surface in N-dimensions (distances), enables parallel
calculations over the grid of points, location of saddle points in the
surface and connecting minima and saddle points
"""
import numpy as np
import itertools as it
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from typing import Dict, Tuple, Union, Optional, Sequence, Iterable
from scipy.interpolate import RectBivariateSpline
from autode.config import Config
from autode.log import logger
from autode.values import ValueArray, Energy
from autode.units import ha, ev, kcalmol, kjmol, J, ang


# Type is a dictionary keyed with tuples and has a set of floats* as a value
_rs_type = Dict[Tuple[int, int], Union[Tuple, np.ndarray]]


class PESnD(ABC):
    """Potential energy surface (PES) in N-dimensions"""

    def __init__(self,
                 species:        Optional['autode.species.Species'] = None,
                 rs:             Optional[_rs_type] = None,
                 allow_rounding: bool = True
                 ):
        """
        N-dimensional PES for a species in a number of distances, defined by
        the 'rs' dictionary containing atom pairs and distances

        -----------------------------------------------------------------------
        Arguments:
            species: Initial species which to evaluate from

            rs: Dictionary of atom indexes (indexed from 0) with associated
                either initial and final, or just final distances. If
                undefined the initial distances are just their current
                values.

            allow_rounding: Allow rounding of a step-size to support an
                            integer number of steps between the initial and
                            final distances
        """
        self._rs = _ListDistances1D(species,
                                    rs_dict=rs if rs is not None else {},
                                    allow_rounding=allow_rounding)

        self._species = species

        self._energies = Energies(np.empty(self.shape), units='Ha')
        self._gradients = np.empty(shape=(*self.shape, self.ndim))
        self._hessians = np.empty(shape=(*self.shape, self.ndim, self.ndim))
        self._init_tensors()

        # Attributes set in calculate()
        self._coordinates: Optional[np.ndarray] = None
        self._method:      Optional['autode.wrappers.base.Method'] = None
        self._n_cores:     int = Config.n_cores
        self._keywords:    Optional['autode.wrappers.keywords.Keywords'] = None

    @property
    def shape(self) -> Tuple:
        """
        Shape of the surface, which is the number of points in each dimension

        -----------------------------------------------------------------------
        Returns:
            (tuple(int)):
        """
        return tuple(len(arr) for arr in self._rs)

    @property
    def ndim(self) -> int:
        """
        Number of dimensions in this PES

        -----------------------------------------------------------------------
        Returns:
            (int):
        """
        return len(self._rs)

    @property
    def origin(self) -> Tuple:
        """
        Tuple of the origin e.g. (0,) in 1D and (0, 0, 0) in 3D

        -----------------------------------------------------------------------
        Returns:
            (tuple(int, ...)):
        """
        return tuple(0 for _ in range(self.ndim))

    def calculate(self,
                  method:   'autode.wrappers.ElectronicStructureMethod',
                  keywords:  Optional['autode.wrappers.Keywords'] = None,
                  n_cores:   Optional[int] = None
                  ) -> None:
        """
        Calculate the surface

        -----------------------------------------------------------------------
        Arguments:
            method: Method to use

            keywords: Keywords to use. If None then will use method.keywords.sp
                      for an unrelaxed or method.keywords.opt for a relaxed

            n_cores: Number of cores. If None then use ade.Config.n_cores
        """
        if self._species is None:
            raise ValueError('Cannot calculate a PES without an initial '
                             'species. Initialise PESNd with a species '
                             'or reactant')

        if keywords is None:
            self._keywords = self._default_keywords(method)
            logger.info('PES calculation keywords not specified, using:\n'
                        f'{self._keywords}')
        else:
            self._keywords = keywords

        # Coordinates tensor is the shape of the PES plus (N, 3) dimensions
        self._coordinates = np.zeros(shape=(*self.shape, self._species.n_atoms, 3),
                                     dtype=np.float64)

        # Set the coordinates of the first point in the PES, and other attrs
        self._coordinates[self.origin] = self._species.coordinates
        self._method = method
        self._n_cores = Config.n_cores if n_cores is None else n_cores

        self._calculate()

        if not self._point_has_energy(self.origin):
            raise RuntimeError('PES calculation failed. Not even the first '
                               'point had an energy')

        return None

    def plot(self,
             filename:      Optional[str] = 'PES.pdf',
             interp_factor: int = 0,
             units:         str = 'kcal mol-1') -> None:
        """
        Plot this PES along a number of dimensions

        -----------------------------------------------------------------------
        Arguments:

            filename: Name of the file to save, type inferred from extension.
                      If None then return .show() on the matplotlib plot

            interp_factor: Factor by which to interpolate the surface with,
                           if 0 (the default) then no interpolation is used

            units: Units of the surface. One of {'Ha', 'eV', 'kcal', 'kJ'}
        """

        if interp_factor < 0:
            raise ValueError(f'Unsupported interpolation factor: '
                             f'{interp_factor}, must be >= 0')

        logger.info(f'Plotting the {self.ndim}D-PES')
        self._set_mpl_params()

        if self.ndim == 1:
            self._plot_1d(interp_factor, units)

        elif self.ndim == 2:
            self._plot_2d(interp_factor, units)

        else:
            raise NotImplementedError(f'Cannot plot a surface in {self.ndim} '
                                      f'dimensions')

        plt.tight_layout()
        plt.subplots_adjust(wspace=0.4 if self.ndim > 1 else None)
        plt.savefig(filename, dpi=500) if filename is not None else plt.show()
        plt.close()
        return None

    def clear(self) -> None:
        """
        Clear the energies, derivatives of, and coordinates on this surface
        """
        return self._init_tensors()

    def save(self, filename: str) -> None:
        """
        Save the PES as a text file (.txt) or compressed numpy file (.npz). If
        .npz then re-loading is possible, either into a PES or otherwise in
        pure numpy. If .txt then will save only the energies in units of
        Hartrees.

        -----------------------------------------------------------------------
        Arguments:
            filename: Name of the file to save. (.txt or .npz) If unknown
                      extension .npz will be added
        """
        if len(self._rs) == 0:
            raise ValueError('Cannot save an empty PES')

        if filename.endswith('.txt'):
            logger.warning('Saving a PES as a .txt file. Not re-loadable')
            arr = np.array(self._energies.to('Ha'))

            if self.ndim > 2:
                logger.warning('Flattening PES to save to .txt file')

            np.savetxt(filename, arr.flatten() if self.ndim > 2 else arr)

        else:
            self._save_npz(filename)

        return None

    def load(self, filename: str) -> None:
        """
        Load a PES from a saved numpy file

        -----------------------------------------------------------------------
        Arguments:
            filename:

        Raises:
            (FileNotFoundError):
        """

        if not filename.endswith('.npz'):
            raise ValueError(f'Cannot reload a PES from {filename}. Must be a '
                             f'.npz compressed numpy file')

        data = np.load(filename, allow_pickle=True)
        self._energies = Energies(data['E'], units='Ha')
        self._coordinates = data['R']

        # Maximum dimension is the largest integer out of e.g. 'r0', 'r1', ...
        ndim = max(int(key.split('r')[1]) for key in data.keys() if 'r' in key)
        self._rs = _ListDistances1D(species=None, rs_dict={})

        for i in range(ndim):
            idx_i, idx_j = tuple(int(idx) for idx in data[f'a{i+1}'])

            self._rs.append(Distances1D(input_array=data[f'r{i+1}'],
                                        atom_idxs=(idx_i, idx_j)))
        return None

    def ts_guesses(self) -> Iterable['autode.species.Species']:
        """
        Generate TS guesses from the saddle points in the energy on this
        surface

        -----------------------------------------------------------------------
        Yields:
            (autode.species.species.Species):
        """
        if not self._point_has_energy(self.origin):
            logger.warning('Initial point on the PES not calculate - have '
                           'no transition state guesses')
            return StopIteration

        for idx, point in enumerate(self._saddle_points()):

            ts_guess = self._species.new_species(name=f'ts_guess{idx}')
            ts_guess.coordinates = self._coordinates[point]

            yield ts_guess

        return StopIteration

    @abstractmethod
    def _default_keywords(self,
                          method: 'autode.wrappers.ElectronicStructureMethod'
                          ) -> 'autode.wrappers.Keywords':
        """
        Default keywords to use for this type of PES e.g. opt or sp

        -----------------------------------------------------------------------
        Arguments:
            method:

        Returns:
            (autode.wrappers.keywords.Keywords):
        """

    @abstractmethod
    def _calculate(self) -> None:
        """Calculate the surface, using method, keywords, n_cores attributes"""

    def _init_tensors(self) -> None:
        """
        Initialise the tensors for the energy (i.e. values on this surface)
        and gradients with respect to each dimension of the surface
        """

        for array in (self._energies, self._gradients, self._hessians):
            array.fill(np.nan)

        if hasattr(self, '_coordinates') and self._coordinates is not None:
            self._coordinates.fill(0.0)

        self._mesh()  # Mesh each coordinate and dynamically add r1, r2.. attrs
        return None

    def _mesh(self) -> None:
        """
        Dynamically add public attributes for r1, r2, ... etc. as nD arrays.
        For example if _rs contains two lists of [0.0, 0.1] then this function
        adds self.r1 and self.r2, each with indexing appropriate for the
        value of r1 at the point (0, 0) on the grid.
        """

        for i, meshed_rs in enumerate(np.meshgrid(*self._rs, indexing='ij')):
            setattr(self, f'r{i+1}', meshed_rs)

        return None

    def _points(self) -> Sequence[Tuple]:
        """
        A list of points in this PES sorted by their sum. For example, for a
        1D PES containing 3 points the list is: [(0,), (1,), (2,)] while for
        a 2D PES of 4 total points the points list is:
        [(0, 0), (0, 1), (1, 0), (1, 1)]
        used for enumerating over the surface in from the initial species
        point (located at the origin).

        -----------------------------------------------------------------------
        Returns:
            (list(tuple(int, ..))): List of points
        """
        ranges = (range(len(r)) for r in self._rs)

        return sorted(it.product(*ranges), key=lambda x: sum(x))

    def _point_name(self, point: Tuple) -> str:
        """
        Name of a particular point in the surface

        -----------------------------------------------------------------------
        Arguments:
            point: Indices of the point

        Returns:
            (str):
        """
        return f'{self._species.name}_scan_{"-".join([str(p) for p in point])}'

    def _is_contained(self, point: Tuple) -> bool:
        """
        Is a point contained on this PES, defined by its indices. For example,
        (-1,) is never on a 1D PES, (2,) is on a 1D PES with 3 points in it
        and (1,) is not on a 2D PES as it doesn't have the same dimension

        -----------------------------------------------------------------------
        Arguments:
            point: Indices of a point on the grid

        Returns:
            (bool): If the point is on the PES
        """
        if len(point) != self.ndim:
            return False

        if sum(point) < 0:
            return False

        if any(p_n >= s_n or p_n < 0 for p_n, s_n in zip(point, self.shape)):
            return False

        return True

    def _point_has_energy(self, point: Tuple) -> bool:
        """
        Does a point have a defined energy? Energies are initialised to
        zero, while failed calculations have np.nan energy.

        -----------------------------------------------------------------------
        Arguments:
            point:

        Returns:
            (bool):
        """
        return (self._is_contained(point)
                and not np.isnan(self._energies[point]))

    def _stationary_points(self,
                           threshold: float = 0.01) -> Sequence[Tuple]:
        """
        Stationary points on the surface, characterised by a zero gradient
        vector. On a finite surface the gradient (g) will never truly vanish,
        so this function will only return those with |g| less than a threshold
        for which all surrounding points have a larger |g|. Is this a
        sufficient condition? who knows.

        -----------------------------------------------------------------------
        Arguments:
            threshold: Maximum |g| (Ha / Å-1) which to consider a
                       stationary point.

        Yields:
            (tuple(int)): Indices of a stationary point
        """
        self._set_gradients()

        for point in self._points():
            grad = self._gradients[point]

            if np.linalg.norm(grad) > threshold:
                continue

            if self._is_minimum_in_gradient(point):
                yield point

        return

    def _saddle_points(self) -> Sequence[Tuple]:
        """
        Find all the saddle points on the surface

        -----------------------------------------------------------------------
        Yields:
            (tuple(int)): Indices of a saddle point
        """
        for point in self._stationary_points():

            self._set_hessian(point)
            hess = self._hessians[point]

            if sum(mu < 0 for mu in np.linalg.eigvals(hess)) == 1:
                yield point

        return StopIteration

    def _set_hessian(self, point: Tuple) -> None:
        """
        Set the Hessian for a particular point in the surface, evaluated
        using finite differences

        -----------------------------------------------------------------------
        Arguments:
            point:
        """
        hessian = self._hessians[point]

        for i in range(self.ndim):
            for j in range(i, self.ndim):

                pm, pp = self._neighbour(point, j, +1), self._neighbour(point, j, -1)

                hessian[i, j] = ((self._gradients[pm][i] - self._gradients[pp][i])
                                 / (self._r(pp, j) - self._r(pm, j)))

                # Hessians are symmetric
                hessian[j, i] = hessian[i, j]

        return None

    def _set_gradients(self) -> None:
        """
        Set the numerical gradient for each point on the surface, in each
        dimension.
        """
        for p in self._points():
            grad = self._gradients[p]   # Gradient with shape: (ndim,)

            if not self._point_has_energy(p):
                logger.warning(f'Cannot set the gradient for point: {p} as it '
                               'did not have an energy')
                grad.fill(np.nan)
                continue

            for n in range(self.ndim):
                pm, pp = self._neighbour(p, n, +1), self._neighbour(p, n, -1)

                if not self._point_has_energy(pm):
                    pm = p

                if not self._point_has_energy(pp):
                    pp = p

                if pm == pp:
                    logger.warning('Cannot determine gradient. Neither '
                                   'neighbour had an energy')
                    grad[n] = np.nan
                    continue

                grad[n] = ((self._energies[pp] - self._energies[pm])
                           / (self._r(pp, n) - self._r(pm, n)))

        return None

    def _is_minimum_in_gradient(self,
                                point:          Tuple) -> bool:
        """
        Is a particular point surrounded by points with larger gradients?
        Only checks ±1 in each dimension, NOT combinations (i.e. diagonals)
        and uses the norm of the gradient (|g|).

        -----------------------------------------------------------------------
        Arguments:
            point:

        Returns:
            (bool):
        """
        norm_grad = np.linalg.norm(self._gradients[point])

        for n in range(self.ndim):
            pm, pp = self._neighbour(point, n, +1), self._neighbour(point, n, -1)

            if not (self._is_contained(pm) and self._is_contained(pp)):
                return False

            for grad in (self._gradients[pm], self._gradients[pp]):

                if np.any(np.isnan(grad)):
                    # Cannot determine if it is a minimum with undefined NN
                    return False

                if np.linalg.norm(grad) < norm_grad:
                    return False

        return True

    @staticmethod
    def _neighbour(point: Tuple, dim: int, delta: int) -> Tuple:
        r"""
        Generate a neighbour of a particular point on the surface. Example::

            Point      dim   delta   -->   Result
            -------------------------------------
            (0,)       0     1              (1,)
            (1, 2)     0     2              (3, 2)
            (1, 2)     1     1              (2, 3)

        -----------------------------------------------------------------------
        Arguments:
            point: Current point

            dim: Dimension in which to find a neighbour

            delta: Order of the neighbour e.g. 1 => nearest neighbour etc.

        Returns:
            (tuple(int, ..)): Point

        Raises:
            (ValueError, IndexError):
        """
        if delta == 0:
            raise ValueError('Cannot find a neighbour using ∆=0 in dimension '
                             f'{dim}')

        new_point = list(point)
        new_point[dim] += delta
        return tuple(new_point)

    def _r(self, point: Tuple, dim: int) -> float:
        """
        Value of r at a particular point, in a given dimension e.g. (0,) in
        a 1D surface in dim 0 -> r_1[0]

        -----------------------------------------------------------------------
        Arguments:
            point: Point on the surface

            dim: Dimension on the surface (indexed from 0)

        Returns:
            (float): r
        """
        idx = point[dim]
        return self._rs[dim][idx]

    def _save_npz(self, filename: str) -> None:
        """Save a compressed numpy array, from which a PES can be re-loaded"""

        if not filename.endswith('.npz'):
            filename += '.npz'

        # Dictionary of flat arrays in each dimension, and their atom indices
        kwds = {f'r{i+1}': np.array(_r) for i, _r in enumerate(self._rs)}
        kwds.update({f'a{i+1}': np.array(_r.atom_idxs, dtype=int)
                     for i, _r in enumerate(self._rs)})

        np.savez(filename,
                 R=self._coordinates,
                 E=np.array(self._energies.to('Ha')),
                 **kwds)

        return None

    def _plot_1d(self, interp_factor: int, units: str) -> None:
        """
        Plot a PES in a single dimension

        -----------------------------------------------------------------------
        Args:
            interp_factor:
            units:
        """
        r_x = self._rs[0]
        energies, units = self._energies, _energy_unit_from_name(units)
        energies = units.conversion * (energies - np.min(energies))

        plt.scatter(r_x, energies,
                    marker='o',
                    s=80,  # Marker size
                    alpha=0.8,  # Opacity
                    zorder=10,  # Order
                    facecolors='white',
                    edgecolors='blue')

        if interp_factor > 0:
            from scipy.interpolate import UnivariateSpline
            spline = UnivariateSpline(r_x, energies)
            r_x = r_x.smoothed(interp_factor)
            energies = spline(r_x)

        # Plot straight lines between the points
        plt.plot(r_x,
                 energies,
                 lw=2,
                 ls='--' if interp_factor > 0 else '-',
                 c='blue',
                 alpha=0.9 if interp_factor > 0 else 0.4)

        plt.ylabel(f'$E$ / {units.plot_name}')
        plt.xlabel('$r$ / Å')

        return None

    def _plot_2d(self, interp_factor: int, units: str) -> None:
        """
        Plot the PES in two dimensions

        -----------------------------------------------------------------------
        Arguments:
            interp_factor:
            units:
        """
        from mpl_toolkits.mplot3d import Axes3D
        from matplotlib.ticker import FormatStrFormatter

        r_x, r_y = self._rs[0], self._rs[1]
        energies = self._energies

        if interp_factor > 0:
            r_x, r_y = r_x.smoothed(interp_factor), r_y.smoothed(interp_factor)
            energies = self._spline_2d()(r_x, r_y)

        # Set up the figure and axes to plot the 3D and projected surfaces on
        _ = plt.figure(figsize=(10, 6))
        ax0 = plt.subplot(1, 2, 1, projection=Axes3D.name)
        ax1 = plt.subplot(1, 2, 2)

        # Convert the energies in the 2D array from the base Hartree units
        units = _energy_unit_from_name(units)
        energies = units.conversion * (energies - np.min(energies))

        ax0.plot_surface(*np.meshgrid(r_x, r_y),
                         energies.T,
                         cmap=plt.get_cmap('plasma'))
        ax0.set_xlabel('$r_1$ / Å')
        ax0.set_ylabel('$r_2$ / Å')
        ax0.set_zlabel(f'$E$ / {units.plot_name}')

        im = ax1.imshow(energies.T,
                        aspect=(r_x.abs_diff / r_y.abs_diff),
                        extent=(r_x[0], r_x[-1],
                                r_y[0], r_y[-1]),
                        origin='lower',
                        cmap=plt.get_cmap('plasma'))

        contour = ax1.contour(*np.meshgrid(r_x, r_y),
                              energies.T,
                              levels=8,
                              origin='lower',
                              colors='k',
                              linewidths=1,
                              alpha=0.5)

        plt.clabel(contour, inline=1, fontsize=10)

        cbar = plt.colorbar(im, fraction=0.0458, pad=0.04)
        cbar.set_label(f'$E$ / {units.plot_name}')
        ax1.set_xlabel('$r_1$ / Å')
        ax1.set_ylabel('$r_2$ / Å')
        ax1.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        ax1.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))

        return None

    @staticmethod
    def _set_mpl_params() -> None:
        """Set some matplotlib (mpl) parameters for nice plotting"""
        import matplotlib as mpl

        mpl.rcParams['axes.labelsize'] = 15
        mpl.rcParams['lines.linewidth'] = 1
        mpl.rcParams['lines.markersize'] = 5
        mpl.rcParams['xtick.labelsize'] = 14
        mpl.rcParams['ytick.labelsize'] = 14
        mpl.rcParams['xtick.direction'] = 'in'
        mpl.rcParams['ytick.direction'] = 'in'
        mpl.rcParams['xtick.top'] = True
        mpl.rcParams['ytick.right'] = True
        mpl.rcParams['axes.linewidth'] = 1.2

        return None

    def _spline_2d(self) -> 'scipy.interpolate.RectBivariateSpline':
        """
        Spline the surface using Scipy. As of scipy v1.7.1 RectBivariateSpline
        can only accept monotonically increasing arrays. This function thus
        reverses arrays and the energies when appropriate, so the spline can
        be fit.

        -----------------------------------------------------------------------
        Returns:
            (scipy.interpolate.RectBivariateSpline): Spline
        """
        r_x, r_y = self._rs[0], self._rs[1]

        if r_x[0] < r_x[-1] and r_y[0] < r_y[-1]:
            # Both x and y are strictly increasing functions
            return RectBivariateSpline(r_x, r_y, self._energies)

        if r_x[0] > r_x[-1] and r_y[0] < r_y[-1]:
            # Swap x order to get strictly increasing in both dims
            return RectBivariateSpline(r_x[::-1], r_y, self._energies[::-1, :])

        if r_x[0] < r_x[-1] and r_y[0] > r_y[-1]:
            # or with y
            return RectBivariateSpline(r_x, r_y[::-1], self._energies[:, ::-1])

        # Reverse both the x and y arrays
        return RectBivariateSpline(r_x[::-1], r_y[::-1],
                                   self._energies[::-1, ::-1])

    def __getitem__(self,
                    indices: Union[Tuple, int]):
        """
        Get a value on this potential energy surface (PES) at a (set of)
        indices

        -----------------------------------------------------------------------
        Arguments:
            indices:

        Returns:
            (autode.values.Energy): Energy
        """
        return Energy(self._energies[indices], units=self._energies.units)

    def __repr__(self):
        return f'PES(shape={self.shape})'


class _ListDistances1D(list):

    def __init__(self, species, rs_dict, allow_rounding=True):
        """Construct a list of distance arrays in each dimension"""
        super().__init__([])

        self._species = species
        self._allow_rounding_of_stepsize = allow_rounding

        for idxs, value in rs_dict.items():
            self.append(self._distance1d_from_key_val(idxs, value))

    def _distance1d_from_key_val(self,
                                 atom_idxs: Tuple[int, int],
                                 value:     Union[Tuple, np.ndarray],
                                 ) -> np.ndarray:
        """
        From a 'value' determine the initial and final distances to use

        -----------------------------------------------------------------------
        Arguments:
            value: Some representation of the final and initial points

        Returns:
            (autode.pes.pes_nd.Distances1D):

        Raises:
            (ValueError): If the value is not of the correct type
        """

        if isinstance(value, tuple):
            return self._distance1d_from_key_val_tuple(atom_idxs, value)

        elif isinstance(value, np.ndarray):
            return Distances1D(value, atom_idxs=atom_idxs)

        else:
            raise ValueError('Unable to populate distance array for atom '
                             f'indices {atom_idxs} with: {value}. Must be '
                             f'either a tuple or numpy array')

    def _distance1d_from_key_val_tuple(self,
                                       atom_idxs: Tuple[int, int],
                                       value:     Union[Tuple[float, Union[float, int]],
                                                        Tuple[float, float, Union[float, int]]]
                                       ):
        """
        Determine a array of distances based on a tuple containing either
        a final distance or a number of steps to perform.

        -----------------------------------------------------------------------
        Arguments:
            atom_idxs: Atom indices
            value:

        Returns:
            (autode.pes.pes_nd.Distances1D):
        """

        if len(value) == 2:

            if self._species is None:
                raise ValueError('Cannot determine initial point without '
                                 'a defined species')

            # Have a pair, the final distance and either the number of steps
            # or the step size
            r_init, r_final = self._species.distance(*atom_idxs), value[0]

        elif len(value) == 3:
            # A triple also defines the initial distance
            r_init, r_final = value[0], value[1]

        else:
            raise ValueError(f'Cannot interpret *{value}* as a final '
                             f'distance and number of steps or step size')

        if isinstance(value[-1], int):
            # Integer values must be a number of steps
            num = value[-1]

        elif isinstance(value[-1], float):
            num = int(round(abs((r_final - r_init) / value[-1]))) + 1

            if not self._allow_rounding_of_stepsize:
                dr = np.sign(r_final - r_init) * abs(value[-1]) * (num - 1)
                r_final = r_init + dr

        else:
            raise ValueError(f'Uninterpretable type: {type(value)}')

        if num <= 1:
            raise ValueError(f'Unsupported number of steps: {num}')

        return Distances1D(np.linspace(r_init, r_final, num=num),
                           atom_idxs=atom_idxs)

    def __eq__(self, other):
        """Equality of two _ListDistances1D instances"""
        return isinstance(other, _ListDistances1D) and super().__eq__(other)


class Distances1D(ValueArray):

    implemented_units = [ang]

    def __new__(cls,
                input_array: Union[np.ndarray, Sequence],
                atom_idxs:   Tuple[int, int]):
        """
        Create an array of distances in a single dimension, with associated
        atom indices, indexed from 0

        -----------------------------------------------------------------------
        Arguments:
            input_array: Array of distances e.g. [1.0, 1.1, 1.2] in Å

            atom_idxs: Indices of the atoms involved in this distance
                       e.g. (0, 1)
        """
        arr = super().__new__(cls, input_array=input_array, units=ang)

        if len(atom_idxs) != 2:
            raise ValueError(f'Indices must be a 2-tuple. Had: {atom_idxs}')

        i, j = atom_idxs
        if not (isinstance(i, int) and isinstance(j, int)):
            raise ValueError(f'Atom indices must be integers. Had: {i}, {j}')

        if i < 0 or j < 0:
            raise ValueError(f'Atom indices must be >0: Had {i}, {j}')

        arr.atom_idxs = atom_idxs
        return arr

    @property
    def min(self) -> float:
        """
        Minimum value of the array

        -----------------------------------------------------------------------
        Returns:
            (float):
        """
        return min(self)

    @property
    def max(self) -> float:
        """
        Maximum value of the array

        -----------------------------------------------------------------------
        Returns:
            (float):
        """
        return max(self)

    @property
    def abs_diff(self) -> float:
        """
        Absolute difference between the minimum and maximum values on this
        array of distances

        Returns:
            (float):
        """
        return abs(self.max - self.min)

    def smoothed(self, factor: int) -> ValueArray:
        """
        Generate a smoothed version of this set of distances, with factor times
        more intermediate points

        -----------------------------------------------------------------------
        Arguments:
            factor: Factor by which to smooth

        Returns:
            (autode.pes.pes_nd.Distances1D):
        """

        new_arr = np.linspace(self.min, self.max, num=factor*len(self))
        return Distances1D(input_array=new_arr, atom_idxs=self.atom_idxs)

    def __repr__(self):
        return f'Distances(n={len(self)}, [{self.min, self.max}])'


class Energies(ValueArray):

    implemented_units = [ha, ev, kcalmol, kjmol, J]

    def __repr__(self):
        """Representation of the energies in a PES"""
        return f'PES{self.ndim}d'


def _energy_unit_from_name(name: str) -> 'autode.units.Unit':
    """
    Generate an energy unit given a name

    ---------------------------------------------------------------------------
    Arguments:
        name: Name of the unit

    Raises:
        (StopIteration): If a suitable energy unit is not found
    """

    for unit in (ha, ev, kcalmol, kjmol, J):
        if name.lower() in unit.aliases:
            return unit

    raise StopIteration(f'Failed to convert {name} to a valid energy unit '
                        f'must be one of: {ha, ev, kcalmol, kjmol, J}')
