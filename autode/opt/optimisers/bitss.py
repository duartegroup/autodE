import numpy as np
from typing import Optional, Union

from autode.log import logger
from autode.values import PotentialEnergy, Distance, GradientRMS
from autode.geom import get_rot_mat_kabsch
from autode.neb import NEB
from autode.opt.optimisers.base import BaseOptimiser
from autode.opt.optimisers.hessian_update import (
    BFGSPDUpdate,
    BFGSDampedUpdate,
    BofillUpdate,
)
from autode.opt.coordinates import CartesianCoordinates, OptCoordinates
from autode.config import Config
from autode.utils import ProcessPool
from autode.opt.optimisers.base import _energy_method_string, method_string
from autode.opt.optimisers.base import _OptimiserHistory
from autode.exceptions import CalculationException, AutodeException
from autode.utils import work_in_tmp_dir
from autode.input_output import atoms_to_xyz_file

import autode.species
import autode.wrappers


class BITSSOptimiser(BaseOptimiser):
    def __init__(
        self,
        initial_species: "autode.species.Species",
        final_species: "autode.species.Species",
        reduction_fac: float = 0.3,
        rmsd_tol_angs: Union[Distance, float] = Distance(0.01, "ang"),
        g_tol: GradientRMS = GradientRMS(1e-3, units="Ha Å-1"),
        max_global_micro_iter: int = 500,
        recalc_constr_freq: int = 30,
        init_trust_radius: float = 0.005,
    ):
        # TODO check reasonableness of microiter values
        # TODO docstrings
        self._history = _OptimiserHistory()
        self._left_img_history = _OptimiserHistory()
        self._right_img_history = _OptimiserHistory()
        self._n_cores = None
        self._method = None

        # TODO should the BinaryImagePair be initialised separately?
        self._imgpair = BinaryImagePair(initial_species, final_species)

        self._maxiter = max_global_micro_iter
        self._reduction_fac = reduction_fac
        n_atoms = self._imgpair._left_image.n_atoms
        dist_tol = Distance(rmsd_tol_angs, "ang") / np.sqrt(n_atoms)
        self._dist_tol = dist_tol
        self._gtol = g_tol
        self._recalc_constr_freq = int(recalc_constr_freq)
        self._trust = init_trust_radius

    @property
    def ts(self) -> Optional["autode.species.Species"]:
        """Return the obtained transition state, if converged"""
        # TODO need further TS opt as well?
        if not self.converged:
            logger.error(
                "BITSS calculation is not converged, "
                "transition state cannot be obtained"
            )
            return None
        n_atoms = self._imgpair._left_image.n_atoms
        rmsd = self._imgpair.euclidean_dist * np.sqrt(n_atoms)
        if rmsd >= 0.1:
            logger.error(
                "BITSS is converged due to loose distance"
                "criteria, however, both images are"
                "significantly different from each other."
                "Returning reactant image"
            )
        return self._imgpair._left_image.copy()

    def calculate(
        self, hmethod=None, lmethod=None, n_cores: int = None
    ) -> None:
        """
        Run the BITSS calculation, until the optimisation criteria
        is reached or the max number of iterations is finished.

        Args:
            hmethod: Method used for calculating energies and gradient
            lmethod: Low-accuracy method, used for estimating Hessian and barrier
            n_cores: Number of cores to use for calculation
        """
        self._imgpair.set_methods_and_n_cores(hmethod, lmethod, n_cores)

        if not self._space_has_degrees_of_freedom:
            logger.info("Optimisation is in a 0D space – terminating")
            return None

        if self._imgpair.euclidean_dist < self._dist_tol:
            logger.error(
                "Reactant and product geometriesa already closer "
                "than provided tolerance - finishing BITSS calcualation"
            )

        logger.error(
            "Starting BITSS calculation to obtain transition state"
            f"with {hmethod} as energies/gradient method and {lmethod} "
            f"for estimating Hessian and current barrier height."
        )

        macroiter_num = 1
        self._coords = self._imgpair.bitss_coords
        self._imgpair.target_dist = (
            1 - self._reduction_fac
        ) * self._imgpair.euclidean_dist
        self.all_update_before_step()
        logger.error(
            f"Macro-iteration: {macroiter_num} - target distance: "
            f"{self._imgpair.target_dist :.3f}"
        )
        logger.error(
            "# iteration    reactant-energy(Ha)    product-energy(Ha)"
            "    BITSS-energy(Ha)    BITSS-grad-RMS(Ha/ang)    "
            "distance(ang)"
        )

        while not self.converged:
            # todo consider the order of things
            # fix this algorithm
            while True:
                # continue RFO microiterations
                self._rfo_step()
                self.all_update_before_step()
                if self._exceeded_maximum_iteration:
                    logger.error("Exceeded the max number of micro-iterations")
                    break
                self._log_convergence()
                if self.rms_grad < 1e-3:
                    break

            if self._exceeded_maximum_iteration:
                logger.error("Exceeded the max number of micro-iterations")
                break

            macroiter_num += 1
            self._imgpair.target_dist = (
                1 - self._reduction_fac
            ) * self._imgpair.target_dist
            logger.error(
                f"Macro-iteration: {macroiter_num} - target distance: "
                f"{self._imgpair.target_dist :.3f}"
            )



        logger.error(
            f"Finished optimisation run - converged: {self.converged}"
        )

    def all_update_before_step(self) -> None:
        """
        All updates that need to be done on the image-pair before
        a geometry step can be taken.
        1. Calculate molecular energies/gradients
        2. Estimate the current barrier (E_B) and update constraints,
        then calculate molecular Hessian
        3. Store BITSS energy in the history (required for plotting)
        3. Calculate the current BITSS gradient
        """
        # TODO the functions must be separate i.e. molecular engrad must not be called
        # within update_bitss_grad()
        self._imgpair.update_molecular_engrad()

        if (
            self._recalc_constr_freq != 0
            and self.iteration % self._recalc_constr_freq == 0
        ):
            self._imgpair.estimate_barrier_and_update_constraints()
            # must recalculate Hessian as underlying potential changes
            self._imgpair.update_molecular_hessian()
            self._imgpair.update_bitss_hessian_by_calculation()
            self._imgpair.update_bitss_grad()
        else:
            self._imgpair.update_bitss_grad()
            self._imgpair.update_bitss_hessian_by_interpolation()
        self._coords.e = self._imgpair.bitss_energy

    @property
    def iteration(self) -> int:
        """
        Iteration of the optimiser, which is equal to the length of the history
        minus one, for zero indexing.

        -----------------------------------------------------------------------
        Returns:
            (int): Current iteration
        """
        return len(self._history) - 1

    @property
    def _space_has_degrees_of_freedom(self) -> bool:
        """Does this optimisation space have any degrees of freedom"""
        return True

    @property
    def _coords(self) -> Optional[OptCoordinates]:
        """
        Current set of coordinates this optimiser is using
        """
        if len(self._history) == 0:
            logger.warning("Optimiser had no history, thus no coordinates")
            return None

        return self._history[-1]

    @_coords.setter
    def _coords(self, value: Optional[OptCoordinates]) -> None:
        """
        Set a new set of coordinates of this optimiser, will append to the
        current history.

        -----------------------------------------------------------------------
        Arguments:
            value (OptCoordinates | None):

        Raises:
            (ValueError): For invalid input
        """
        if value is None:
            return
        elif isinstance(value, OptCoordinates):
            self._history.append(value.copy())
        else:
            raise ValueError(
                f"Cannot set the optimiser coordinates with {value}"
            )

    def _rfo_step(self) -> None:
        """
        Take an RFO step with this image-pair. Should only act on
        self._imgpair._coords
        """
        h_n, _ = self._imgpair.hess.shape
        # Form the augmented Hessian, structure from ref [1], eqn. (56)
        aug_H = np.zeros(shape=(h_n + 1, h_n + 1))
        aug_H[:h_n, :h_n] = self._imgpair.hess
        aug_H[-1, :h_n] = self._imgpair.grad
        aug_H[:h_n, -1] = self._imgpair.grad

        aug_H_lmda, aug_H_v = np.linalg.eigh(aug_H)
        # A RF step uses the eigenvector corresponding to the lowest non-zero
        # eigenvalue
        mode = np.where(np.abs(aug_H_lmda) > 1e-16)[0][0]

        # and the step scaled by the final element of the eigenvector
        delta_s = aug_H_v[:-1, mode] / aug_H_v[-1, mode]

        self._take_step_within_trust_radius(delta_s)
        return None

    def _take_step_within_trust_radius(
        self, delta_s: np.ndarray, factor: float = 1.0
    ) -> float:
        """
        Update the coordinates while ensuring the step isn't too large in
        cartesian coordinates

        -----------------------------------------------------------------------
        Arguments:
            delta_s: Step in Cartesian coordinates

        Returns:
            factor: The coefficient of the step taken
        """
        logger.error("Taking RFO step")

        if len(delta_s) == 0:  # No need to sanitise a null step
            return 0.0

        cartesian_delta = factor * delta_s
        new_coords = self._imgpair.bitss_coords + cartesian_delta
        delta_magnitude = np.linalg.norm(cartesian_delta)

        if delta_magnitude > self._trust:
            logger.error(
                f"Calculated step is too large ({delta_magnitude:.3f} Å)"
                f" - scaling down"
            )

            factor = self._trust / delta_magnitude
            new_coords = (
                    self._imgpair.bitss_coords
                    + self._trust / delta_magnitude * delta_s
            )

        self._imgpair.bitss_coords = new_coords
        self._coords = new_coords  # also update local history
        return factor

    @property
    def rms_grad(self) -> GradientRMS:
        """RMS of current BITSS gradient"""
        return GradientRMS(np.sqrt(np.average(np.square(self._imgpair.grad))))

    @property
    def converged(self) -> bool:
        """Has this optimisation converged"""
        # todo maybe do iteration == 0
        return (
            self._imgpair.euclidean_dist < self._dist_tol
            and self.rms_grad < self._gtol
        )

    @property
    def last_energy_change(self) -> PotentialEnergy:
        """Last ∆E found in this"""

        if self.iteration > 0:
            delta_e = self._history.final.e - self._history.penultimate.e
            return PotentialEnergy(delta_e, units="Ha")

        if self.converged:
            logger.warning(
                "Optimiser was converged in less than two "
                "cycles. Assuming an energy change of 0"
            )
            return PotentialEnergy(0)

        return PotentialEnergy(np.inf)

    @property
    def final_coordinates(self) -> Optional["autode.opt.OptCoordinates"]:
        return None if len(self._history) == 0 else self._history.final

    def _log_convergence(self) -> None:
        """Log the iterations in the form:
        Iteration   |∆E| / kcal mol-1    ||∇E|| / Ha Å-1
        """
        logger.error(
            f"#iteration:{self.iteration:4}   reactant:{self._imgpair._left_image.energy:.6f}"
            f"    product:{self._imgpair._right_image.energy:.6f}   "
            f"  BITSS energy:{self._imgpair.bitss_energy:.3f}  "
            f"  RMS BITSS gradient:{self.rms_grad:.3f} "
            f"  Distance:{self._imgpair.euclidean_dist:.3f}  "
            f"  Target distance:{self._imgpair.target_dist:.3f}"
        )

    @property
    def _exceeded_maximum_iteration(self) -> bool:
        """
        Has this optimiser exceeded the maximum number of iterations
        allowed?

        -----------------------------------------------------------------------
        Returns:
            (bool):
        """
        if self.iteration >= self._maxiter:
            logger.warning(
                f"Reached the maximum number of iterations "
                f"*{self._maxiter}*. Did not converge"
            )
            return True

        else:
            return False


class BinaryImagePair:
    """
    A pair of images (species) separated by a minimum-energy path.
    Used for BITSS optimisation.

    All calculations are done in units of Hartree and Angstrom for
    consistency
    """

    def __init__(
        self,
        initial_species: autode.Species,
        final_species: autode.Species,
        alpha: float = 10.0,
        beta: float = 0.01,  # todo check different values of beta
    ):
        # todo ensure that unit is consistent (especially arrays)
        # todo set different names here, so that new_species not needed for grad, hess etc.
        assert isinstance(initial_species, autode.Species)
        assert isinstance(final_species, autode.Species)
        logger.warning(
            "Any constraints, conformers or previous gradient/hessian/energy\n"
            "in initial_species and final_species will be ignored."
        )
        self._left_image = initial_species.new_species("initial_species")
        self._right_image = final_species.new_species("final_species")
        self._left_image.coordinates = self._left_image.coordinates.to("ang")
        self._right_image.coordinates = self._right_image.coordinates.to("ang")
        self._sanity_check()
        self._align_species()

        self._left_history = _OptimiserHistory()
        self._right_history = _OptimiserHistory()

        self._engrad_method = None
        self._hess_barrier_method = None
        self._n_cores = None

        self.kappa_eng = None  # kappa_e
        self.kappa_dist = None  # kappa_d
        self.estimated_barrier = None  # E_B
        self.alpha = alpha
        self.beta = beta
        self._target_dist = None

        self.grad = None  # current gradient
        self.hess = None  # current hessian
        self.last_grad = None  # previous gradient
        self.last_hess = None  # previous Hessian
        self.last_bitss_coords = None
        self._hessian_update_types = [
            BFGSPDUpdate,
            BFGSDampedUpdate,
            BofillUpdate,
        ]

    def _sanity_check(self) -> None:
        """
        Check if the starting and ending points have the same
        charge, multiplicity and the same atoms in the same order
        """

        if self._left_image.n_atoms != self._right_image.n_atoms:
            raise ValueError(
                "The initial_species and final_species must "
                "have the same number of atoms: "
                f"{self._left_image.n_atoms} != "
                f"{self._right_image.n_atoms}"
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
                    "both species"
                )

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

    def set_methods_and_n_cores(
        self,
        hmethod: "autode.wrappers.methods.Method",
        lmethod: "autode.wrappers.methods.Method",
        n_cores: int,
    ) -> None:
        """
        Sets methods and n_cores for all calculations run on
        this BinaryImagePair.

        Args:
            hmethod: The method used to calculate gradients and energies
            lmethod: The method used to calculate hessians and estimated
                barrier required for calculating constraints (kappa)
            n_cores: Max number of cores for all calculations
        """
        # TODO calculate hessian should only calculate hessian, not grad
        from autode.wrappers.methods import Method

        if isinstance(hmethod, Method):
            self._engrad_method = hmethod
        else:
            raise ValueError(
                "The supplied hmethod has to be of type"
                "autode.wrappers.methods.Method, but "
                f"{type(hmethod)} was supplied"
            )
        if isinstance(lmethod, Method):
            self._hess_barrier_method = lmethod
        else:
            raise ValueError(
                "The supplied lmethod has to be of type"
                "autode.wrappers.methods.Method, but "
                f"{type(lmethod)} was supplied"
            )
        self._n_cores = int(n_cores)

    def update_molecular_engrad(self) -> None:
        """
        Update the gradient and energies of both molecules in
        the image pair, using supplied hmethod
        """
        logger.error(
            f"Calculating molecular en/grad with {self._engrad_method}"
        )
        n_cores_pp = max(self._n_cores // 2, 1)
        n_workers = 2 if 2 < self._n_cores else self._n_cores
        with ProcessPool(max_workers=n_workers) as pool:
            jobs = [
                pool.submit(
                    _calculate_en_grad_for_species,
                    species=mol.copy(),
                    method=self._engrad_method,
                    n_cores=n_cores_pp,
                )
                for idx, mol in enumerate(
                    [self._left_image, self._right_image]
                )
            ]
            (self._left_image.energy,
             self._left_image.gradient) = jobs[0].result()
            (self._right_image.energy,
             self._right_image.gradient) = jobs[1].result()

    def update_bitss_grad(self) -> None:
        """
        Update the BITSS gradient from the molecular gradients
        and distances and constraints, by analytic formula
        """
        assert self._left_image.gradient is not None
        assert self._right_image.gradient is not None
        logger.error(
            "Calculating gradient of BITSS energy against all"
            "coordinates, using analytic differentiation"
        )

        left_grad = np.array(self._left_image.gradient.flatten())
        right_grad = np.array(self._right_image.gradient.flatten())
        left_energy = float(self._left_image.energy)
        right_energy = float(self._right_image.energy)
        # energy terms from coordinates of reactant (x_1)
        # del E_1 * (1 + 2 * kappa_e * (E_1 - E_2))
        left_coords_term = left_grad * (
            1 + 2 * self.kappa_eng * (left_energy - right_energy)
        )
        # energy terms from coordinates of product (x_2)
        # del E_2 * (1 + 2 * kappa_e * (E_2 - E_1))  # notice the flipped energies
        right_coords_term = right_grad * (
            1 + 2 * self.kappa_eng * (right_energy - left_energy)
        )
        # distance term
        # del d * 2 * kappa_d * (d - d_i)
        dist_vec = (
            self._left_image.coordinates.flatten()
            - self._right_image.coordinates.flatten()
        )
        # todo update with A matrix
        distance_grad = (
            2
            * (1 / self.euclidean_dist)
            * np.concatenate((dist_vec, -dist_vec))
            * self.kappa_dist
            * (self.euclidean_dist - self.target_dist)
        )
        # total grad
        self.grad = (
            np.concatenate((left_coords_term, right_coords_term))
            + distance_grad
        )
        # todo remove these parts
        rms_dist_grad = np.sqrt(np.average(np.square(distance_grad)))
        energy_grad = np.sqrt(
            np.average(
                np.square(
                    np.concatenate((left_coords_term, right_coords_term))
                )
            )
        )
        print("RMS of distance grad = ", rms_dist_grad)
        print("RMS of energy grad = ", energy_grad)
        print("max dist grad = ", np.max(distance_grad))
        print(
            "max energy grad = ",
            np.max(np.concatenate((left_coords_term, right_coords_term))),
        )

    def update_molecular_hessian(self) -> None:
        """
        Updates the Hessian for the molecules of this image pair
        with the supplied lmethod
        """
        n_cores_pp = max(self._n_cores // 2, 1)
        n_workers = 2 if 2 < self._n_cores else self._n_cores
        logger.error(
            "Calculating Hessian for both ends of BinaryImagePair "
            f"with {self._hess_barrier_method}"
        )
        with ProcessPool(max_workers=n_workers) as pool:
            jobs = [
                pool.submit(
                    _calculate_hessian_for_species,
                    species=mol,
                    method=self._hess_barrier_method,
                    n_cores=n_cores_pp,
                )
                for idx, mol in enumerate(
                    [self._left_image, self._right_image]
                )
            ]
            self._left_image.hessian = jobs[0].result()
            self._right_image.hessian = jobs[1].result()

    def update_bitss_hessian_by_calculation(self) -> None:
        """
        Calculate the Hessian of BITSS energy, using low-level
        molecular Hessian and gradients and energies
        """
        # TODO question: should the gradient be of lmethod or hmethod? If think of hessian
        # as approximation, then hmethod is fine. Paper does not say anything on this.
        assert self._left_image.hessian is not None
        assert self._right_image.hessian is not None

        logger.error(
            "Calculating Hessian of BITSS energy against all "
            "coordinates, with analytic differentiations"
        )

        left_hess = np.array(self._left_image.hessian)
        right_hess = np.array(self._right_image.hessian)
        left_grad = np.array(self._left_image.gradient.flatten())
        right_grad = np.array(self._right_image.gradient.flatten())
        left_energy = float(self._left_image.energy)
        right_energy = float(self._right_image.energy)

        # terms from E_1, E_2 in upper left square of Hessian
        upper_left_sq = left_hess * (
            1
            + 2 * left_energy * self.kappa_eng
            - 2 * right_energy * self.kappa_eng
        )
        upper_left_sq += (
            2
            * self.kappa_eng
            * (left_grad.reshape(-1, 1) @ left_grad.reshape(1, -1))
        )
        # terms from E_1, E_2 in lower right square of Hessian
        lower_right_sq = right_hess * (
            1
            - 2 * left_energy * self.kappa_eng
            + 2 * right_energy * self.kappa_eng
        )
        lower_right_sq += (
            2
            * self.kappa_eng
            * (right_grad.reshape(-1, 1) @ right_grad.reshape(1, -1))
        )
        # terms from E_1, E_2 in upper right square of Hessian
        upper_right_sq = (
            -2
            * self.kappa_eng
            * (left_grad.reshape(-1, 1) @ right_grad.reshape(1, -1))
        )
        # terms from E_1, E_2 in lower left square of Hessian
        lower_left_sq = (
            -2
            * self.kappa_eng
            * (right_grad.reshape(-1, 1) @ left_grad.reshape(1, -1))
        )

        # put together all energy terms
        upper_part = np.hstack((upper_left_sq, upper_right_sq))
        lower_part = np.hstack((lower_left_sq, lower_right_sq))
        energy_hess = np.vstack((upper_part, lower_part))
        # distance terms
        # del d * 2 * kappa_d * (d - d_i)
        dist_vec = (
            self._left_image.coordinates.flatten()
            - self._right_image.coordinates.flatten()
        )
        distance_grad = (
            2
            * (1 / self.euclidean_dist)
            * np.concatenate((dist_vec, -dist_vec))
            * self.kappa_dist
            * (self.euclidean_dist - self.target_dist)
        )
        hess_d = (1 / self.euclidean_dist) * (
            self.A_mat()
            - (distance_grad.reshape(-1, 1) @ distance_grad.reshape(1, -1))
        )
        distance_hess = (
            2
            * self.kappa_dist
            * (distance_grad.reshape(-1, 1) @ distance_grad.reshape(1, -1))
        )
        distance_hess += (
            2
            * self.kappa_dist
            * self.euclidean_dist
            * (1 - 2 * self.target_dist)
            * hess_d
        )
        self.hess = energy_hess + distance_hess
        self._make_hessian_positive_definite()

    def initialise_run(self):
        pass

    def A_mat(self):
        num_atoms = self._left_image.n_atoms * 3
        return np.vstack(
            (
                np.hstack((np.identity(num_atoms), -np.identity(num_atoms))),
                np.hstack((-np.identity(num_atoms), np.identity(num_atoms))),
            )
        )

    def _calculate_estimated_barrier(self, num_images: int = 10):
        """
        Estimate the current energy barrier by running a linear
        interpolation, and taking the highest point's energy minus
        the average of energy of two endpoints
        """
        assert self._hess_barrier_method is not None
        logger.error(
            f"Using a linear interpolation of {num_images} to "
            f"estimate current barrier"
        )

        # Use NEB to interpolate a linear path with 10 points (default)
        linear_path = NEB(
            self._left_image.copy(), self._right_image.copy(), num=num_images
        )
        n_cores_pp = max(self._n_cores // num_images, 1)
        n_workers = num_images if num_images < self._n_cores else self._n_cores
        # TODO check the logic of n_worker and n_cores_pp
        with ProcessPool(max_workers=n_workers) as pool:
            jobs = [
                pool.submit(
                    _calculate_sp_energy_for_species,
                    species=image.species.new_species(name=f"img{idx}"),
                    method=self._hess_barrier_method,
                    n_cores=n_cores_pp,
                )
                for idx, image in enumerate(linear_path.images)
            ]
            path_energies = [job.result() for job in jobs]
        # E_B = max(interpolated E's) - avg(reactant , product)
        estimated_barrier = (
            max(path_energies) - (path_energies[0] + path_energies[-1]) / 2
        )
        self.estimated_barrier = float(estimated_barrier.to("Ha"))

    def estimate_barrier_and_update_constraints(self):
        """
        Estimate the energy barrier and then use it, along with
        gradient information to update constraints.
        """
        assert self._left_image.gradient is not None
        assert self._right_image.gradient is not None

        self._calculate_estimated_barrier()
        logger.error(
            "Updating BITSS energy and distance constraints"
            "using estimated barrier and calculated gradients"
        )

        # kappa_e = alpha / (2 * E_B)
        self.kappa_eng = float(self.alpha / (2 * self.estimated_barrier))

        left_grad = np.array(self._left_image.gradient.flatten())
        right_grad = np.array(self._right_image.gradient.flatten())
        # kappa_d = max(
        # sqrt(|grad E_1|^2 + |grad E_2|^2) / (2 * sqrt(2) * beta * d_i),
        # E_B / (beta * d_i^2)
        # )
        # grad E_1 and grad E_2 must be projected in the direction of d_i
        dist_vec = (
            self._left_image.coordinates.flatten()
            - self._right_image.coordinates.flatten()
        )
        grad_left_proj = abs(np.dot(left_grad, dist_vec)) / np.linalg.norm(
            dist_vec
        )
        grad_right_proj = abs(np.dot(right_grad, dist_vec)) / np.linalg.norm(
            dist_vec
        )
        # todo test projected value vs real value

        kappa_d_first_option = np.sqrt(
            grad_left_proj**2 + grad_right_proj**2
        ) / (2 * np.sqrt(2) * self.beta * self.euclidean_dist)
        kappa_d_second_option = self.estimated_barrier / (
            self.beta * self.euclidean_dist**2
        )

        self.kappa_dist = float(
            max(kappa_d_first_option, kappa_d_second_option)
        )

    def update_bitss_hessian_by_interpolation(self):
        """
        Updates the hessian by interpolation using one of the
        Hessian update formulas, requires the current and old
        BITSS gradient, and old BITSS hessian
        """
        for update_type in self._hessian_update_types:
            updater = update_type(
                h=self.last_hess,
                s=self.bitss_coords.raw - self.last_bitss_coords.raw,
                y=self.grad - self.last_grad,
                subspace_idxs=self.bitss_coords.indexes,
            )

            if not updater.conditions_met:
                logger.warning(
                    f"Hessian update not possible "
                    f"with {update_type}, retrying"
                )
            else:
                break

        logger.error(f"Updating Hessian with {updater}")
        self.hess = updater.updated_h
        self._make_hessian_positive_definite()
        # todo remove
        eigvals = np.linalg.eigvalsh(self.hess)
        print("Highest eigenvalue = ", max(eigvals))

    @property
    def bitss_energy(self) -> PotentialEnergy:
        """
        The current BITSS energy
        """
        energy = self._left_image.energy + self._right_image.energy
        energy += (
            self.kappa_eng
            * (self._left_image.energy - self._right_image.energy) ** 2
        )
        energy += (
            self.kappa_dist
            * float((self.euclidean_dist - self.target_dist)) ** 2
        )
        return energy

    @property
    def bitss_coords(self) -> CartesianCoordinates:
        """
        Current coordinates for this Binary Image-Pair

        Returns:
            (CartesianCoordinates)
        """
        return CartesianCoordinates(
            np.concatenate(
                (
                    self._left_image.coordinates.flatten(),
                    self._right_image.coordinates.flatten(),
                )
            )
        ).copy()  # copy to make it non-editable

    @bitss_coords.setter
    def bitss_coords(self, value: CartesianCoordinates) -> None:
        """
        Update the coordinates of the current image pair. Changes
        the coordinates of both reactant and product

        Args:
            value (CartesianCoordinates): Flattened coordinates of all atoms
        """
        # TODO the first iteration is not present in left_history or right_history?
        if value is None:
            return
        if isinstance(value, np.ndarray):
            assert value.shape == (
                2 * 3 * self._left_image.n_atoms,
            ), "Provided array does not have the right shape"
        else:
            raise ValueError(
                f"bitss_coords only accepts arrays but {type(value)}"
                " was provided"
            )
        # store only coordinates and energy
        left_coords = CartesianCoordinates(
            self._left_image.coordinates.flatten()
        )
        left_coords.e = (
            self._left_image.energy.copy()
        )  # todo setting wrong energies?
        right_coords = CartesianCoordinates(
            self._right_image.coordinates.flatten()
        )
        right_coords.e = self._right_image.energy.copy()
        self._left_history.append(left_coords.copy())
        self._right_history.append(right_coords.copy())
        self.last_bitss_coords = CartesianCoordinates(
            np.concatenate((left_coords, right_coords))
        ).copy()
        self._left_image.coordinates = value[
            : 3 * self._left_image.n_atoms
        ].copy()
        self._right_image.coordinates = value[
            3 * self._right_image.n_atoms :
        ].copy()
        # update all properties
        self.last_grad = self.grad
        self.grad = None
        self.last_hess = self.hess
        self.hess = None

    @property
    def euclidean_dist(self) -> Distance:
        """
        The Euclidean distance between the reactant and product
        of this image-pair

        Returns:
            (Distance): Distance in Angstrom
        """
        return Distance(
            np.sqrt(
                np.sum(
                    np.square(
                        self._left_image.coordinates
                        - self._right_image.coordinates
                    )
                )
            ),
            units="ang",
        )

    @property
    def target_dist(self) -> Distance:
        """
        The target distance in the current macro-iteration

        Returns:
            (Distance)
        """
        return self._target_dist.to("ang")

    @target_dist.setter
    def target_dist(self, value: Distance) -> None:
        """
        Set the target distance for the current macro-iteration

        Args:
            value (Distance):
        """
        if value is None:
            return
        if isinstance(value, Distance):
            print("setter called")
            self._target_dist = value.to("ang")
        else:
            raise ValueError

    def _make_hessian_positive_definite(
            self, min_eigenvalue: float = 1e-5
    ) -> None:
        """
        Ensure that the eigenvalues of a matrix are all >0 i.e. the matrix
        is positive definite. Will shift all values below min_eigenvalue to that
        value.

        ---------------------------------------------------------------------------
        Arguments:
            min_eigenvalue: Minimum value eigenvalue of the matrix
        """

        if self.hess is None:
            raise RuntimeError(
                "Cannot make a positive definite Hessian, "
                "there is no Hessian"
            )

        lmd, v = np.linalg.eig(self.hess)  # Eigenvalues and eigenvectors

        if np.all(lmd > min_eigenvalue):
            logger.info("Hessian was positive definite")
            return

        logger.warning(
            "Hessian was not positive definite. "
            "Shifting eigenvalues and reconstructing"
        )
        lmd[lmd < min_eigenvalue] = min_eigenvalue
        self.hess = np.linalg.multi_dot((v, np.diag(lmd), v.T)).real
        return

    def _write_single_trajectory(self, side):
        """Writes the trajectory of any one species into xyz"""
        if side == "left":
            tmp_image = self._left_image.copy()
            hist = self._left_history
        elif side == "right":
            tmp_image = self._right_image.copy()
            hist = self._right_history
        else:
            return
        # todo check if file exists
        for coord in hist:
            tmp_image.coordinates = coord
            fname = tmp_image.name + ".trj.xyz"
            atoms_to_xyz_file(tmp_image.atoms, fname, append=True)

    def write_left_trajectory(self):
        self._write_single_trajectory("left")

    def write_right_trajectory(self):
        self._write_single_trajectory("right")

    def plot_energies(self, side):
        import matplotlib.pyplot as plt
        if side == "left":
            hist = self._left_history
            name = self._left_image.name
        elif side == "right":
            hist = self._right_history
            name = self._right_image.name
        else:
            return
        energies = []
        for coord in hist:
            eng = coord.e
            if eng is None:
                raise Exception
            energies.append(eng)
        plt.plot(energies,'.')
        plt.show()  #name+"_energy_plot.pdf")


def _calculate_sp_energy_for_species(
    species: autode.Species,
    method: "autode.wrappers.methods.Method",
    n_cores: int,
):
    """
    Convenience function for calculating the single point
    energy for a molecule; removes all output and input files
    for the calculation
    """
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
    return species.energy


@work_in_tmp_dir()
def _calculate_hessian_for_species(
    species, method: "autode.wrappers.methods.Method", n_cores
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
    hess_calc.clean_up(force=True)
    return species.hessian.to("ha/ang^2")


def _calculate_en_grad_for_species(species, method, n_cores):
    """
    Convenience function for calculating energy and gradient for
    a molecule; removes all input and output files for the
    calculation
    """
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
