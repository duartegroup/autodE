from abc import ABC, abstractmethod
from typing import Optional, List, Tuple
import numpy as np

from autode.values import Distance, PotentialEnergy, Gradient
from autode.geom import get_rot_mat_kabsch
from autode.opt.coordinates import OptCoordinates, CartesianCoordinates
from autode.opt.optimisers.hessian_update import BofillUpdate, NullUpdate
from autode.opt.optimisers.base import _OptimiserHistory
from autode.utils import work_in_tmp_dir
from autode.log import logger

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


def _calculate_energy_for_species(
    species: autode.species.species.Species,
    method: autode.wrappers.methods.Method,
    n_cores: int,
):
    from autode.calculations import Calculation

    sp_calc = Calculation(
        name=f"{species.name}_sp",
        molecule=species,
        method=method,
        keywords=method.keywords.sp,
        n_cores=n_cores,
    )
    sp_calc.run()
    sp_calc.clean_up(force=True, everything=True)
    return species.energy.to('Ha')


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
        Initialize the image pair, does not set methods/n_cores or
        hessian update types!

        Args:
            left_image: One molecule of the pair
            right_image: Another molecule of the pair
        """
        assert isinstance(left_image, autode.species.species.Species)
        assert isinstance(right_image, autode.species.species.Species)
        self._left_image = left_image.new_species(name="left_image")
        self._right_image = right_image.new_species(name="right_image")
        self._sanity_check()
        self._align_species()

        # separate methods for engrad and hessian calc
        self._engrad_method = None
        self._hess_method = None
        self._n_cores = None
        # Bofill has no conditions, so kept as default
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

    def _align_species(self) -> None:
        """
        Translates both molecules to origin and then performs
        a Kabsch rotation to orient the molecules as close as
        possible against each other
        """
        # first translate the molecules to the origin
        logger.error(
            "Translating initial_species (reactant) "
            "and final_species (product) to origin"
        )
        p_mat = self._left_image.coordinates.copy()
        p_mat -= np.average(p_mat, axis=0)
        self._left_image.coordinates = p_mat

        q_mat = self._right_image.coordinates.copy()
        q_mat -= np.average(q_mat, axis=0)
        self._right_image.coordinates = q_mat

        logger.error(
            "Rotating initial_species (reactant) "
            "to align with final_species (product) "
            "as much as possible"
        )
        rot_mat = get_rot_mat_kabsch(p_mat, q_mat)
        rotated_p_mat = np.dot(rot_mat, p_mat.T).T
        self._left_image.coordinates = rotated_p_mat

    def set_method_and_n_cores(
        self,
        engrad_method: autode.wrappers.methods.Method,
        n_cores: int,
        hess_method: Optional[autode.wrappers.methods.Method] = None,
    ) -> None:
        """
        Sets the methods for engrad and hessian calculation, and the
        total number of cores used for any calculation in this image pair

        Args:
            engrad_method (autode.wrappers.methods.Method):
            n_cores (int):
            hess_method (autode.wrappers.methods.Method|None):
        """
        if not isinstance(engrad_method, autode.wrappers.methods.Method):
            raise ValueError(
                f"The engrad_method needs to be of type autode."
                f"wrappers.method.Method, But "
                f"{type(engrad_method)} was supplied."
            )
        self._engrad_method = engrad_method
        if hess_method is None:
            pass
        elif not isinstance(hess_method, autode.wrappers.methods.Method):
            raise ValueError(
                f"The hess_method needs to be of type autode."
                f"wrappers.method.Method, But "
                f"{type(hess_method)} was supplied."
            )
        self._hess_method = hess_method
        self._n_cores = int(n_cores)
        return None

    @property
    def n_atoms(self) -> int:
        """Number of atoms"""
        return self._left_image.n_atoms

    @property
    def total_iters(self) -> int:
        """Total number of iterations done on this image pair"""
        return len(self._left_history) + len(self._right_history) - 2

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

    def get_coord_by_side(self, side: str) -> CartesianCoordinates:
        """For external usage, supplies only the coordinate object"""
        _, coord, _, _ = self._get_img_by_side(side)
        return coord

    def _get_img_by_side(
        self, side: str
    ) -> Tuple[autode.Species, CartesianCoordinates, _OptimiserHistory, float]:
        """
        Access an image and some properties by a string that
        represents side. Returns a tuple of the species, the
        current coordinate object, and a factor that is necessary
        for calculation

        Args:
            side (str): 'left' or 'right'

        Returns:
            (tuple) : tuple(image, current coord, history, fac)
        """
        if side == "left":
            img = self._left_image
            coord = self.left_coord
            hist = self._left_history
            fac = 1.0
        elif side == "right":
            img = self._right_image
            coord = self.right_coord
            hist = self._right_history
            fac = -1.0
        else:
            raise Exception

        return img, coord, hist, fac

    def update_one_img_molecular_engrad(self, side: str) -> None:
        """
        Update the molecular energy and gradient using the supplied
        engrad_method for one image only

        Args:
            side (str): 'left' or 'right'
        """
        assert self._engrad_method is not None
        assert self._n_cores is not None
        img, coord, _, _ = self._get_img_by_side(side)

        logger.error(f"Calculating engrad for {side} side"
                     f" with {self._engrad_method}")
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
        img, coord, _, _ = self._get_img_by_side(side)

        logger.error(f"Calculating Hessian for {side} side"
                     f" with {self._hess_method}")
        hess = _calculate_hessian_for_species(
            species=img.copy(), method=self._hess_method, n_cores=self._n_cores
        )
        img.hessian = hess
        coord.update_h_from_cart_h(hess)
        return None

    def update_one_img_molecular_hessian_by_formula(self, side: str) -> None:
        img, coord, hist, _ = self._get_img_by_side(side)
        assert len(hist) > 1, "Hessian update not possible!"
        assert coord.h is None, "Hessian already exists!"
        last_coord = hist.penultimate
        for update_type in self._hessian_update_types:
            updater = update_type(
                h=last_coord.h,
                s=coord.raw - last_coord.raw,
                y=coord.g - last_coord.g,
                subspace_idxs=coord.indexes,
            )
            if not updater.conditions_met:
                continue

            coord.update_h_from_cart_h(updater.updated_h)
            break

        return None
