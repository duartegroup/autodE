from abc import ABC, abstractmethod
from typing import Optional, List, Tuple
import numpy as np

from autode.values import Distance, PotentialEnergy, Gradient
from autode.opt.coordinates import OptCoordinates, CartesianCoordinates
from autode.opt.optimisers.hessian_update import BofillUpdate
from autode.opt.optimisers.base import _OptimiserHistory
from autode.utils import work_in_tmp_dir

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


class BaseImagePair:
    """
    Base class for a pair of images (e.g., reactant and product) of
    the same species. The images are called 'left' and 'right' to
    distinguish them, but there is no requirement for one to be
    reactant or product.
    """

    def __init__(
        self,
        left_image: autode.species.species.Species,
        right_image: autode.species.species.Species,
    ):
        """
        Initialize the image pair, does not set methods/n_cores!

        Args:
            left_image: One molecule of the pair
            right_image: Another molecule of the pair
        """
        assert isinstance(left_image, autode.species.species.Species)
        assert isinstance(right_image, autode.species.species.Species)
        self._left_image = left_image.new_species(name="left_image")
        self._right_image = right_image.new_species(name="right_image")
        self._sanity_check()

        # separate methods for engrad and hessian calc
        self._engrad_method = None
        self._hess_method = None
        self._n_cores = None
        self._hessian_update_types = [BofillUpdate]

        self._left_history = _OptimiserHistory()
        self._right_history = _OptimiserHistory()
        # push the first coordinates into history
        self.left_coord = self._left_image.coordinates.to("ang").flatten()
        self.right_coord = self._right_image.coordinates.to("ang").flatten()

    def _sanity_check(self) -> None:
        """
        Check if the two supplied images have the same
        charge, multiplicity and the same atoms in the same order
        """

        if self._left_image.n_atoms != self._right_image.n_atoms:
            raise ValueError(
                "The initial_species and final_species must "
                "have the same number of atoms!"
            )

        if (
            self._left_image.charge != self._right_image.charge
            or self._left_image.mult != self._right_image.mult
        ):
            raise ValueError(
                "Charge/multiplicity of initial_species and "
                "final_species supplied are not the same"
            )

        for idx in range(len(self._left_image.atoms)):
            if (
                self._left_image.atoms[idx].label
                != self._right_image.atoms[idx].label
            ):
                raise ValueError(
                    "The order of atoms in initial_species "
                    "and final_species must be the same. The "
                    f"atom at position {idx} is different in"
                    "the two species"
                )

        return None

    def set_method_and_n_cores(
        self,
        engrad_method: autode.wrappers.methods.Method,
        hess_method: autode.wrappers.methods.Method,
        n_cores: int,
    ) -> None:
        """
        Sets the methods for engrad and hessian calculation, and the
        total number of cores used for any calculation in this image pair

        Args:
            engrad_method (autode.wrappers.methods.Method):
            hess_method (autode.wrappers.methods.Method):
            n_cores (int):
        """
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
        return None

    @property
    def n_atoms(self):
        """Number of atoms"""
        return self._left_image.n_atoms

    @property
    def left_coord(self) -> Optional[CartesianCoordinates]:
        """The coordinates of the left image"""
        if len(self._left_history) == 0:
            return None
        return self._left_history[-1]

    @left_coord.setter
    def left_coord(self, value):
        """
        Sets the coordinates of the left image, also updates
        the coordinates of the species

        Args:
            value (np.ndarray|CartesianCoordinates): flat or (n_atoms, 3) array
                                                     of coordinates
        """
        if value is None:
            return
        elif isinstance(value, CartesianCoordinates):
            self._left_history.append(value.copy())
        elif (
            isinstance(value, np.ndarray)
            and value.flatten().shape[0] == 3 * self.n_atoms
        ):
            self._left_history.append(CartesianCoordinates(value))
        else:
            raise ValueError
        self._left_image.coordinates = self._left_history[-1]

    @property
    def right_coord(self) -> Optional[CartesianCoordinates]:
        """The coordinates of the right image"""
        if len(self._right_history) == 0:
            return None
        return self._right_history[-1]

    @right_coord.setter
    def right_coord(self, value):
        """
        Sets the coordinates of the left image, also updates
        the coordinates of the species

        Args:
            value (np.ndarray|CartesianCoordinates): flat or (n_atoms, 3) array
                                                     of coordinates
        """
        if value is None:
            return
        elif isinstance(value, CartesianCoordinates):
            self._right_history.append(value.copy())
        elif (
            isinstance(value, np.ndarray)
            and value.flatten().shape[0] == 3 * self.n_atoms
        ):
            self._right_history.append(CartesianCoordinates(value))
        else:
            raise ValueError
        self._right_image.coordinates = self._right_history[-1]

    def _get_img_by_side(
        self, side: str
    ) -> Tuple[autode.Species, CartesianCoordinates, float]:
        """
        Access an image and some properties by a string that
        represents side. Returns a tuple of the species, the
        current coordinate object, and a factor that is necessary
        for calculation

        Args:
            side (str): 'left' or 'right'

        Returns:
            tuple(autode.Species, CartesianCoordinates, float):
        """
        if side == "left":
            img = self._left_image
            coord = self.left_coord
            fac = 1.0
        elif side == "right":
            img = self._right_image
            coord = self.right_coord
            fac = -1.0
        else:
            raise Exception

        return img, coord, fac

    def update_one_img_molecular_engrad(self, side: str) -> None:
        """
        Update the molecular energy and gradient using the supplied
        engrad_method for one image only

        Args:
            side (str): 'left' or 'right'
        """
        assert self._engrad_method is not None
        assert self._n_cores is not None
        img, coord, _ = self._get_img_by_side(side)

        en, grad = _calculate_engrad_for_species(
            species=img.copy(),
            method=self._engrad_method,
            n_cores=self._n_cores,
        )
        # update both species and coord
        img.energy = en
        img.gradient = grad.copy()
        coord.e = en.copy()
        coord.update_g_from_cart_g(grad)
        return None

    def update_one_img_molecular_hessian_by_calc(self, side: str) -> None:
        """
        Updates the molecular hessian using supplied hess_method
        for one image only

        Args:
            side (str): 'left' or 'right'

        Returns:

        """
        assert self._hess_method is not None
        assert self._n_cores is not None
        img, coord, _ = self._get_img_by_side(side)

        hess = _calculate_hessian_for_species(
            species=img.copy(), method=self._hess_method, n_cores=self._n_cores
        )
        img.hessian = hess
        coord.update_h_from_cart_h(hess)
        return None

    def update_one_img_molecular_hessian_by_formula(self, side: str) -> None:
        img, coord, _ = self._get_img_by_side(side)
        # todo


class DistanceConstrainedImagePair(BaseImagePair):
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
        super().__init__(left_image, right_image)

        # C_d = (d - d_i)
        self._d_i = None
        self._lambda_dist = 0  # dist. constraint lagrange multiplier

    def get_one_img_jacobian_of_constraints(self, side: str) -> np.ndarray:
        """
        Obtains the Jacobian (first derivative) of the constraints
        on one image of the image pair (i.e. only one side considered)

        Args:
            side (str): 'left' or 'right'

        Returns:
            (np.ndarray): An (n_atoms, 1) shaped matrix,
                          in units of Hartree/angs
        """
        _, _, fac = self._get_img_by_side(side)
        # 1st column is derivatives of C_E (if exists)
        # 2nd column is derivatives of C_d
        dist_vec = np.array(self.left_coord - self.right_coord)
        # grad(d - d_i) = grad(d)
        # grad_1(d) = (1/d) (r_1 - r_2)
        # grad_2(d) = (1/d) (r_2 - r_1) = - (1/d) (r_1 - r_2)
        a_matrix = (
            float(1 / self.euclid_dist) * fac * dist_vec.reshape(-1, 1)
        )  # column vector, gradient of distance constraint

        return a_matrix

    def get_one_img_lagrangian_gradient(self, side: str) -> np.ndarray:
        """
        Build the gradient of the Lagrangian, for one image only

        Args:
            side (str): 'left' or 'right'

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
        a_matrix = self.get_one_img_jacobian_of_constraints(side=side)
        grad_con = grad - (a_matrix @ lmda_col)

        # grad(L) = [[grad_con, -C_d]]
        grad_L = np.vstack((grad_con, constr_func_col))
        return grad_L.flatten()

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
        a_matrix = self.get_one_img_jacobian_of_constraints(side=side)
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
        # hess(L) = [[W, -A],
        #            [-A.T, 0]]
        lagrange_hess = np.vstack(
            (
                np.hstack((w_matrix, -a_matrix)),
                np.hstack((-a_matrix.T, end_zero)),
            )
        )

        return lagrange_hess

    @property
    def n_constraints(self) -> int:
        return 1

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
    def C_d(self) -> float:
        """
        Returns the current value of the dustance constraint function
        (d - d_i) in Hartree
        """
        return float(self.euclid_dist.to("ang") - self.target_dist.to("ang"))

    def update_lagrangian_multiplier(self, num: float):
        """
        Updates the Lagrangian multiplier lambda_dist
        """
        lambda_val = float(num)
        self._lambda_dist = lambda_val
