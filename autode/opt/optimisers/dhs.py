from typing import Optional, List, Tuple
import numpy as np

from autode.values import Distance, PotentialEnergy, Gradient
from autode.opt.coordinates import OptCoordinates, CartesianCoordinates
from autode.opt.optimisers.base import _OptimiserHistory
from autode.utils import work_in_tmp_dir, ProcessPool
from autode.log import logger
from autode.config import Config

import autode.species.species
import autode.wrappers.methods


def _calculate_engrad_for_species(
    species: autode.species.species.Species,
    method: autode.wrappers.methods.Method,
    n_cores: int,
):
    from autode.calculations import Calculation

    engrad_calc = Calculation(
        name=f"{species.name}_engrad",
        molecule=species,
        method=method,
        keywords=method.keywords.grad,
        n_cores=n_cores,
    )
    engrad_calc.run()
    engrad_calc.clean_up(force=True, everything=True)
    return species.energy.to("Ha"), species.gradient.to("Ha/ang")


@work_in_tmp_dir()
def _calculate_hessian_for_species(
    species: autode.species.species.Species,
    method: autode.wrappers.methods.Method,
    n_cores: int,
):
    """
    Convenience function for calculating the Hessian for a
    molecule; removes all input and output files for the
    calculation
    """
    from autode.calculations import Calculation

    hess_calc = Calculation(
        name=f"{species.name}_hess",
        molecule=species,
        method=method,
        keywords=method.keywords.hess,
        n_cores=n_cores,
    )
    hess_calc.run()
    hess_calc.clean_up(force=True, everything=True)
    return species.hessian.to("ha/ang^2")


class DistanceConstrainedImagePair:
    """
    A pair of images (reactant and product) of the same species,
    constrained by Euclidean distance between their coordinates
    (d = d_i), where d_i is a constant. The constraint is enforced
    by Lagrangian multipliers
    """
    def __init__(
        self,
        left_image: autode.species.species.Species,
        right_image: autode.species.species.Species,
    ):
        """
        Initialize the constrained image pair. Does not initialize
        methods!

        Args:
            left_image: One molecule of the pair
            right_image: Another molecule of the pair
        """
        assert isinstance(left_image, autode.species.species.Species)
        assert isinstance(right_image, autode.species.species.Species)
        self._left_image = left_image.new_species(name="left_image")
        self._right_image = right_image.new_species(name="right_image")
        # todo sanity check that two images have same charge etc. and they are not
        # too close

        self._engrad_method = None
        self._hess_method = None
        self._n_cores = None

        # C_d = (d - d_i)
        self._d_i = None
        self._lambda_dist = 0  # dist. constraint lagrange multiplier

        self._left_history = _OptimiserHistory()
        self._right_history = _OptimiserHistory()
        self.left_coord = self._left_image.coordinates.to('ang').flatten()
        self.right_coord = self._right_image.coordinates.to('ang').flatten()

    def set_method_and_n_cores(
        self,
        engrad_method: autode.wrappers.methods.Method,
        hess_method: autode.wrappers.methods.Method,
        n_cores: int,
    ):
        if not isinstance(engrad_method, autode.wrappers.methods.Method):
            raise ValueError(
                f"The engrad_method needs to be of type autode."
                f"wrappers.method.Method, But "
                f"{type(engrad_method)} was supplied."
            )
        self._engrad_method = engrad_method
        if not isinstance(hess_method, autode.wrappers.methods.Method):
            raise ValueError(
                f"The hess_method needs to be of type autode."
                f"wrappers.method.Method, But "
                f"{type(hess_method)} was supplied."
            )
        self._hess_method = hess_method
        self._n_cores = int(n_cores)

    def update_one_side_molecular_engrad(self, side: str):
        assert self._engrad_method is not None
        assert self._n_cores is not None
        if side == "left":
            img = self._left_image
            coord = self.left_coord
        elif side == "right":
            img = self._right_image
            coord = self.right_coord
        else:
            raise Exception

        en, grad = _calculate_engrad_for_species(
            species=img.copy(),
            method=self._engrad_method,
            n_cores=self._n_cores,
        )
        img.energy = en
        img.gradient = grad.copy()
        coord.update_g_from_cart_g(grad)

    def get_one_sided_jacobian_of_constraints(self, side: str) -> np.ndarray:
        """
        Obtains the Jacobian (first derivative) of the constraints
        on one side of the image pair (i.e. only one side considered)

        Args:
            side: 'left' or 'right'

        Returns:
            (np.ndarray): An (n_atoms * n_constraints) shaped matrix,
                          in units of Hartree/angs
        """
        if side == "left":
            img = self._left_image
            fac = 1.0
        elif side == "right":
            img = self._right_image
            fac = -1.0
        else:
            raise Exception
        # 1st column is derivatives of C_E (if exists)
        # 2nd column is derivatives of C_d
        dist_vec = np.array(
            self._left_image.coordinates.to("ang").flatten()
            - self._right_image.coordinates.to("ang").flatten()
        )
        # grad(d - d_i) = grad(d)
        # grad_1(d) = (1/d) (r_1 - r_2)
        # grad_2(d) = (1/d) (r_2 - r_1) = - (1/d) (r_1 - r_2)
        dist_constr_grad = (
            float(1 / self.euclid_dist) * fac * dist_vec.reshape(-1, 1)
        )  # column vector
        a_matrix = dist_constr_grad

        return a_matrix

    def get_one_sided_lagrangian_gradient(self, side: str) -> np.ndarray:
        """
        Get the gradient of the Lagrangian, for one side only

        Args:
            side: 'left' or 'right'

        Returns:
            (np.ndarray): The gradient of Lagrangian (L) in a flat array
        """
        if side == "left":
            img = self._left_image
        elif side == "right":
            img = self._right_image
        else:
            raise Exception
        assert img.gradient is not None

        grad = np.array(img.gradient.to("ha/ang")).flatten().reshape(-1, 1)
        # g_con = g - A @ lambda  <= lambda is a column matrix of multipliers
        lmda_col = np.array([self._lambda_dist]).reshape(-1, 1)
        constr_func_col = np.array([-self.C_d]).reshape(-1, 1)
        a_matrix = self.get_one_sided_jacobian_of_constraints(side=side)
        grad_con = grad - (a_matrix @ lmda_col)

        # grad(L) = [[grad_con, -C_d]]
        grad_L = np.vstack((grad_con, constr_func_col))
        return grad_L.flatten()

    def update_one_side_molecular_hessian(self, side: str) -> None:
        assert self._hess_method is not None
        assert self._n_cores is not None
        if side == "left":
            img = self._left_image
            coord = self.left_coord
        elif side == "right":
            img = self._right_image
            coord = self.right_coord
        else:
            raise Exception

        hess = _calculate_hessian_for_species(
            species=img.copy(), method=self._hess_method, n_cores=self._n_cores
        )
        img.hessian = hess
        coord.update_h_from_cart_h(hess)
        return None

    def get_one_sided_hessians_of_constraints(
        self, side: str
    ) -> List[np.ndarray]:
        """
        Obtain the Hessians of the constraint functions C_E and C_d

        Args:
            side: 'left' or 'right'

        Returns:
            (List[np.ndarray]): Hessian of distance constraint
        """
        # A list is returned so that this class can be subclassed
        # for energy constraints
        if side == "left":
            img = self._left_image
            fac = 1.0
        elif side == "right":
            img = self._right_image
            fac = -1.0
        else:
            raise Exception

        hess_list = []
        # hess(d - d_i) = hess(d)
        dist_vec = np.array(
            self._left_image.coordinates.to("ang").flatten()
            - self._right_image.coordinates.to("ang").flatten()
        )
        grad_d = (
            float(1 / self.euclid_dist) * fac * dist_vec.reshape(-1, 1)
        )  # column vector
        hess_d = float(1 / self.euclid_dist) * (
            np.identity(self.n_atoms) - (grad_d @ grad_d.T)
        )
        hess_list.append(hess_d)

        return hess_list

    def get_one_sided_lagrangian_hessian(self, side: str) -> np.ndarray:
        if side == "left":
            img = self._left_image
        elif side == "right":
            img = self._right_image
        else:
            raise Exception
        # todo use coord.h instead of img.hessian
        assert img.hessian is not None
        h_matrix = np.array(img.hessian.to("ha/ang^2"))
        constr_hessians = self.get_one_sided_hessians_of_constraints(side=side)
        a_matrix = self.get_one_sided_jacobian_of_constraints(side=side)
        return self._form_lagrange_hessian(h_matrix, constr_hessians, a_matrix)

    def _form_lagrange_hessian(
        self,
        h_matrix: np.ndarray,
        constr_hessians: List[np.ndarray],
        a_matrix: np.ndarray,
    ):
        """
        Forms the Hessian of the Lagrangian from its components

        Args:
            h_matrix: Unconstrained hessian of system
            constr_hessians: Hessians of constraint functions
            a_matrix: Jacobian of constraint functions

        Returns:
            (np.ndarray): Hessian of Lagrangian. Square matrix with
                          dimensions of (n_atoms + n_constraints)
        """
        w_matrix = h_matrix - self._lambda_dist * constr_hessians[0]

        end_zero = np.zeros((self.n_constraints, self.n_constraints))
        # hess(L) = [[W, -A],[-A.T, 0]]
        lagrange_hess = np.vstack(
            (
                np.hstack((w_matrix, -a_matrix)),
                np.hstack((-a_matrix.T, end_zero)),
            )
        )

        return lagrange_hess

    @property
    def n_atoms(self):
        """Number of atoms"""
        return self._left_image.n_atoms

    @property
    def n_constraints(self) -> int:
        return 2 if self._is_e_constr else 1

    @property
    def left_coord(self) -> Optional[CartesianCoordinates]:
        if len(self._left_history) == 0:
            return None
        return self._left_history[-1]

    @left_coord.setter
    def left_coord(self, value):
        if value is None:
            return
        elif isinstance(value, CartesianCoordinates):
            self._left_history.append(value.copy())
        elif isinstance(value, np.ndarray) and value.shape == 3 * self.n_atoms:
            self._right_history.append(CartesianCoordinates(value))
        else:
            raise ValueError
        self._left_image.coordinates = self._left_history[-1]

    @property
    def right_coord(self) -> Optional[CartesianCoordinates]:
        if len(self._right_history) == 0:
            return None
        return self._right_history[-1]

    @right_coord.setter
    def right_coord(self, value):
        if value is None:
            return
        elif isinstance(value, CartesianCoordinates):
            self._right_history.append(value.copy())
        elif isinstance(value, np.ndarray) and value.shape == 3 * self.n_atoms:
            self._right_history.append(CartesianCoordinates(value))
        else:
            raise ValueError
        self._right_image.coordinates = self._right_history[-1]

    @property
    def target_dist(self) -> Distance:
        return Distance(self._d_i, units="ang")

    @target_dist.setter
    def target_dist(self, value):
        if isinstance(value, Distance):
            self._d_i = value.to("ang")
        elif isinstance(value, float):
            self._d_i = Distance(value, units="ang")
        else:
            raise ValueError(
                "The value of target_dist must be either"
                " autode.values.Distance or float, but "
                f"{type(value)} was supplied"
            )

    @property
    def euclid_dist(self) -> Distance:
        """
        Returns the euclidean distance between the two images of the pair
        """
        dist_vec = np.array(
            self._left_image.coordinates.to("ang").flatten()
            - self._right_image.coordinates.to("ang").flatten()
        )
        return Distance(np.linalg.norm(dist_vec), units="ang")

    @property
    def C_E(self) -> float:
        """
        Returns the current value of the energy constraint function
        (E_1 - E_2) in Hartree
        """
        return float(
            self._left_image.energy.to("Ha")
            - self._right_image.energy.to("Ha")
        )

    @property
    def C_d(self) -> float:
        """
        Returns the current value of the dustance constraint function
        (d - d_i) in Hartree
        """
        return float(self.euclid_dist.to("ang") - self.target_dist.to("ang"))

    def update_lagrangian_multipliers(self, num):
        """
        Updates the Lagrangian multiplier lambda_dist
        """
        lambda_val = float(num)
        self._lambda_dist = lambda_val


_optional_method = Optional[autode.wrappers.methods.Method]


class DHS:
    def __init__(
        self,
        initial_species: autode.species.Species,
        final_species: autode.species.Species,
        maxiter: int = 10,
        reduction_factor: float = 0.05,
        dist_tol: float = 0.05,
    ):
        self.imgpair = DistanceConstrainedImagePair(
            initial_species, final_species
        )
        self._reduction_fac = reduction_factor

        if int(maxiter) <= 0:
            raise ValueError(
                "An optimiser must be able to run at least one "
                f"step, but tried to set maxiter = {maxiter}"
            )

        self._maxiter = int(maxiter)
        self._dist_tol = Distance(dist_tol, units="ang")
        self._engrad_method = None
        self._hess_method = None
        self._n_cores = None

        self._grad = None
        self._hess = None

        # todo have a history where store the optimised points after each macroiter

    @property
    def macroiter_converged(self):
        if self.imgpair.euclid_dist <= self._dist_tol:
            return True
        else:
            return False

    @property
    def microiter_converged(self):
        # gtol and dist_tol
        pass

    @property
    def _exceeded_maximum_iteration(self) -> bool:
        return True if self.imgpair.total_iters > self._maxiter else False

    def calculate(
        self,
        engrad_method: _optional_method = None,
        hess_method: _optional_method = None,
        n_cores: Optional[int] = None,
    ):
        from autode.methods import (
            method_or_default_hmethod,
            method_or_default_lmethod,
        )

        engrad_method = method_or_default_hmethod(engrad_method)
        hess_method = method_or_default_lmethod(hess_method)
        if n_cores is None:
            n_cores = Config.n_cores
        self.imgpair.set_method_and_n_cores(
            engrad_method=engrad_method,
            hess_method=hess_method,
            n_cores=n_cores,
        )

        self._initialise_run()

        # in each macroiter, the distance criteria is reduced by factor
        macroiter_num = 0
        maxiter_reached = False
        while not self.macroiter_converged:
            self.imgpair.target_dist = self.imgpair.euclid_dist * (
                1 - self._reduction_fac
            )
            macroiter_num += 1
            while not self.microiter_converged:
                if self.imgpair.left_coord.e < self.imgpair.right_coord.e:
                    side = "left"
                else:
                    side = "right"
                self._take_one_sided_step(side)
                self._update_one_side_mol_engrad_hess(side)
                # in gradient update step, update hessian as well
                # no need to update hessian in prfo step
                if self._exceeded_maximum_iteration:
                    # error message
                    maxiter_reached = True
                    break
            if maxiter_reached:
                break

        pass

    def _take_one_sided_step(self, side: str):
        grad = self.imgpair.get_one_sided_lagrangian_gradient(side)
        hess = self.imgpair.get_one_sided_lagrangian_hessian(side)
        self._prfo_step(side, grad, hess)

    def _update_one_side_mol_engrad_hess(self, side: str):
        self.imgpair.update_one_side_molecular_engrad(side)
        # todo interpolate hessian by Bofill
        #self.imgpair.update_one_side_molecular_hessian(side)

    def _initialise_run(self):
        # todo this func is empty
        # todo is it really necessary to parallelise in DHS?
        self.imgpair.update_one_side_molecular_engrad('left')
        self.imgpair.update_one_side_molecular_hessian('left')
        self.imgpair.update_one_side_molecular_engrad('right')
        self.imgpair.update_one_side_molecular_hessian('right')
        pass

    def _prfo_step(self, side: str, grad: np.ndarray, hess:np.ndarray):
        b, u = np.linalg.eigh(hess)
        f = u.T.dot(grad)

        delta_s = np.zeros(shape=(self.imgpair.n_atoms,))

        # partition the RFO
        # The constraint is the last item
        constr_components = u[-1, :]
        # highest component of constraint must be the required mode
        constr_mode = np.argmax(constr_components)
        u_max = u[:, constr_mode]
        b_max = b[constr_mode]
        f_max = f[constr_mode]

        # only one constraint - distance
        aug_h_max = np.zeros(shape=(2, 2))
        aug_h_max[0, 0] = b_max
        aug_h_max[1, 0] = f_max
        aug_h_max[0, 1] = f_max
        lambda_p = np.linalg.eigvalsh(aug_h_max)[-1]

        delta_s -= f_max * u_max / (b_max - lambda_p)

        u_min = np.delete(u, constr_mode, axis=1)
        b_min = np.delete(b, constr_mode)
        f_min = np.delete(f, constr_mode)
        # todo deal with rot./trans.?

        # n_atoms non-constraint modes
        m = self.imgpair.n_atoms
        aug_h_min = np.zeros(shape=(m + 1, m + 1))
        aug_h_min[:m, :m] = np.diag(b_min)
        aug_h_min[-1, :m] = f_min
        aug_h_min[:m, -1] = f_min
        eigvals_min = np.linalg.eigvalsh(aug_h_min)
        min_mode = np.where(np.abs(eigvals_min) > 1e-15)[0][0]
        lambda_n = eigvals_min[min_mode]

        for i in range(m):
            delta_s -= f_min[i] * u_min[:, i] / (b_min[i] - lambda_n)

        self.take_step_within_trust_radius(side, delta_s)

    def take_step_within_trust_radius(self, side, delta_s):
        # set the coords to one side
        pass
