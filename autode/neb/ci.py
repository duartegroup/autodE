"""
Climbing image (CI) nudged elastic band implementation from
https://doi.org/10.1063/1.1329672
"""
import numpy as np
from autode.neb.original import NEB, Images, Image
from autode.log import logger


class CImage(Image):

    def get_force(self, im_l, im_r):
        """
        Compute F_m

        Arguments:
            im_l (autode.neb.Image): Left image (i-1)
            im_r (autode.neb.Image): Right image (i+1)
        """
        # τ,  x_i-1,  x_i,   x_i+1
        hat_tau, x_l, x, x_r = self._tau_xl_x_xr(im_l, im_r)

        # F_m = ∇V(x_m) + (2∇V(x_m).τ)τ
        return -self.grad + 2.0 * np.dot(self.grad, hat_tau) * hat_tau

    def __init__(self, image):
        """Construct a climbing image from a non-climbing one

        Arguments:
            image (autode.neb.Image):
        """
        super().__init__(image.name, image.k)
        # Set all the current attributes from the regular image
        self.__dict__.update(image.__dict__)


class CImages(Images):

    def increment(self):
        """Increment the counter, and switch on a climbing image"""
        super().increment()

        if self[0].iteration < 4:
            # No need to do anything else
            return None

        if self.peak_idx is None:
            logger.error('Lost NEB peak - cannot switch on CI')
            return None

        logger.info(f'Setting image {self.peak_idx} as the CI')
        self[self.peak_idx] = CImage(image=self[self.peak_idx])
        return None

    def __init__(self, images):
        """Initalise a set of images """

        super().__init__(num=len(images), init_k=0)
        for i, image in enumerate(images):
            self[i] = image


class CINEB(NEB):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.images = CImages(self.images)
