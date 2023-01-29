from typing import Optional
import numpy as np

from autode.values import Distance
from autode.opt.coordinates import OptCoordinates
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


class ConstrainedImagePair:
    """
    A pair of images (reactant and product) constrined by
    distance (d = d_i) and energy (E_1 = E_2) criteria.
    The constraints are enforced by Lagrangian multipliers
    """

    def __init__(
        self,
        left_image: autode.species.species.Species,
        right_image: autode.species.species.Species,
        constrain_energy: bool = True,
    ):
        """
        Initialize the constrained image pair. Does not initialize
        methods!

        Args:
            left_image: One molecule of the pair
            right_image: Another molecule of the pair
            constrain_energy (bool): Are energy constrained applied
        """
        assert isinstance(left_image, autode.species.species.Species)
        assert isinstance(right_image, autode.species.species.Species)
        self._left_image = left_image
        self._right_image = right_image
        # todo sanity check that two images have same charge etc. and they are not
        # too close
        self._is_e_constr = bool(constrain_energy)

        self._engrad_method = None
        self._hess_method = None
        self._n_cores = None

        # C_E = (E_1 - E_2)
        # C_d = (d - d_i)
        self._d_i = None
        self._lambda_eng = 0  # energy constraint lagrange multiplier
        self._lambda_dist = 0  # dist. constraint lagrange multiplier

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

    def update_one_side_molecular_grad(self, side: str):
        assert self._engrad_method is not None
        assert self._n_cores is not None
        if side == "left":
            image = self._left_image
        elif side == "right":
            image = self._right_image
        else:
            raise Exception

        en, grad = _calculate_engrad_for_species(
            species=image, method=self._engrad_method, n_cores=self._n_cores
        )
        image.energy = en
        image.gradient = grad

    def get_combined_jacobian_of_constraints(self) -> np.ndarray:
        """
        Obtains the Jacobian (first dervatives) of the constraints
        for the combined image pair (i.e. both ends considered)

        Returns:
             (np.ndarray): An (2 n_atoms * n_constraints) shaped array,
                           in units of Hartree/Angstrom
        """
        # A_matrix is n_atoms * n_constraints matrix
        # 1st column is derivatives of C_E against coords (if present)
        # 2nd column is derivatives of C_d against coords
        dist_vec = np.array(
            self._left_image.coordinates.to("ang").flatten()
            - self._right_image.coordinates.to("ang").flatten()
        )
        # grad(d-d_i) = grad(d) = (1/d) * [(r_1 - r_2), (r_2 - r_1)]
        dist_constr_grad = (1 / self.euclid_dist) * np.concatenate(
            (dist_vec, -dist_vec)
        ).reshape(-1, 1)
        A_matrix = dist_constr_grad
        if self._is_e_constr:
            assert self._left_image.gradient is not None
            assert self._right_image.gradient is not None
            # grad(E_1-E_2) = [grad(E_1), -grad(E_2)]
            eng_constr_grad = np.concatenate(
                (
                    np.array(self._left_image.gradient.to("ha/ang")),
                    -np.array(self._right_image.gradient.to("ha/ang")),
                )
            ).reshape(-1, 1)
            A_matrix = np.hstack((dist_constr_grad, eng_constr_grad))
        return A_matrix

    def get_one_sided_jacobian_of_constraints(self, side: str) -> np.ndarray:
        """
        Obtains the Jacobian (first derivative) of the constraints
        on one side of the image pair (i.e. only one side considered)

        Returns:
            (np.ndarray): An (n_atoms * n_constraints) shaped matrix
        """
        if side == "left":
            img = self._left_image
            fac = 1
        elif side == "right":
            img = self._right_image
            fac = -1
        else:
            raise Exception
        # 1st column is derivatives of C_E (if exists)
        # 2nd column is derivatives of C_d
        dist_vec = np.array(
            self._left_image.coordinates.to("ang").flatten()
            - self._right_image.coordinates.to("ang").flatten()
        )
        # grad(d - d_i) = grad(d)
        dist_constr_grad = (
            float(1 / self.euclid_dist) * fac * dist_vec.reshape(-1, 1)
        )  # column vector
        A_matrix = dist_constr_grad
        if self._is_e_constr:
            assert self._left_image.gradient is not None
            assert self._right_image.gradient is not None
            # grad(E_1-E_2) = grad(E_1) if 1 else -grad(E_2)
            eng_constr_grad = fac * (
                np.array(img.gradient.to("ha/ang")).reshape(-1, 1)
            )
            A_matrix = np.hstack((dist_constr_grad, eng_constr_grad))
        return A_matrix
        # todo use this to get the two sided jacobian

    def update_both_side_molecular_grad(self):
        # todo parallelise
        pass

    def get_left_lagrangian_gradient(self):
        #
        pass

    def update_left_hessian(self):
        pass

    def update_right_hessian(self):
        pass

    @property
    def n_constraints(self):
        return 2 if self._is_e_constr else 1

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


class DHS:
    def __init__(
        self,
        initial_species: autode.species.Species,
        final_species: autode.species.Species,
        reduction_factor: float = 0.05,
    ):
        self.imgpair = ConstrainedImagePair(
            initial_species, final_species, constrain_energy=False
        )
        self._reduction_fac = reduction_factor
