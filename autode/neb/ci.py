"""
Climbing image (CI) nudged elastic band implementation from
https://doi.org/10.1063/1.1329672
"""
import numpy as np
from typing import Optional
from autode.neb.original import NEB, Images, Image
from autode.log import logger


class CImage(Image):
    def __init__(self, image):
        """
        Construct a climbing image from a non-climbing one

        -----------------------------------------------------------------------
        Arguments:
            image (autode.neb.Image):
        """
        super().__init__(image.name, image.k)
        # Set all the current attributes from the regular image
        self.__dict__.update(image.__dict__)

    def get_force(self, im_l, im_r) -> np.ndarray:
        """
        Compute F_m

        -----------------------------------------------------------------------
        Arguments:
            im_l (autode.neb.Image): Left image (i-1)
            im_r (autode.neb.Image): Right image (i+1)
        """
        # τ,  x_i-1,  x_i,   x_i+1
        hat_tau, x_l, x, x_r = self._tau_xl_x_xr(im_l, im_r)

        # F_m = ∇V(x_m) + (2∇V(x_m).τ)τ
        return -self.grad + 2.0 * np.dot(self.grad, hat_tau) * hat_tau


class CImages(Images):
    def __init__(
        self,
        images: Images,
        num: int = None,
        wait_iterations: int = 4,
        init_k: Optional[float] = None,
    ):
        """
        Initialise a set of images

        ----------------------------------------------------------------------
        Arguments:
            images (autode.neb.Images):

            wait_iterations (int): Number of iterations to wait before turning
                                  on the climbing image

            init_k: Initial force constant
        """
        super().__init__(
            num=num if num is not None else len(images),
            init_k=init_k if init_k is not None else images[0].k,
        )

        self.wait_iteration = wait_iterations
        for i, image in enumerate(images):
            self[i] = image

    def __eq__(self, other):
        """Equality of climbing image NEB"""
        if (
            not isinstance(other, CImages)
            or self.wait_iteration != other.wait_iteration
        ):
            return False

        return super().__eq__(other)

    def increment(self) -> None:
        """Increment the counter, and switch on a climbing image"""
        super().increment()

        if self[0].iteration < self.wait_iteration:
            # No need to do anything else
            return

        if self.peak_idx is None:
            logger.error("Lost NEB peak - cannot switch on CI")
            return

        logger.info(f"Setting image {self.peak_idx} as the CI")
        self[self.peak_idx] = CImage(image=self[self.peak_idx])
        return None


class CINEB(NEB):

    _images_type = CImages

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.images = CImages(self.images)

    def _minimise(
        self, method, n_cores, etol, max_n=30
    ) -> "scipy.optimize.OptimizeResult":
        """Minimise th energy of every image in the NEB"""
        logger.info(f"Minimising to ∆E < {etol:.4f} Ha on all NEB coordinates")
        result = super()._minimise(method, n_cores, etol, max_n)

        if any(
            im.iteration > self.images.wait_iteration for im in self.images
        ):
            return result

        logger.info(
            "Converged before CI was turned on. Reducing the wait and "
            "minimising again"
        )

        self.images.wait_iteration = max(im.iteration for im in self.images)
        result = super()._minimise(method, n_cores, etol, max_n)

        return result
