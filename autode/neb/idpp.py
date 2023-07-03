import numpy as np

from scipy.spatial import distance_matrix
from typing import TYPE_CHECKING

from autode.values import PotentialEnergy

if TYPE_CHECKING:
    from autode.neb.original import Image, Images


class IDPP:
    """
    Image dependent pair potential (IDPP) objective function from
    https://arxiv.org/pdf/1406.1512.pdf

    .. math::

        S = Σ_i  Σ_{j>i} w(r_{ij}) (r_{ij}^{(k)} - r_{ij})^2

    where :math:`r_{ij}` is the distance between atoms i and j and
    :math:`r_{ij}^{(k)} = r_{ij}^{(1)} + k(r_{ij}^{(N)} - r_{ij}^{(1)})/N` for
    :math:`N` images. The weight function is :math:`w(r_{ij}) = r_{ij}^{-4}`,
    as suggested in the paper.
    """

    def __init__(self, images: "Images"):
        """Initialise a IDPP potential from a set of NEB images"""

        if len(images) < 2:
            raise ValueError("Must have at least 2 images for IDPP")

        # Distance matrices containing all r_{ij}^k
        self._dists = {image_k.name: None for image_k in images}
        self._diagonal_distance_matrix_idxs = None

        self._set_distance_matrices(images)

    def __call__(self, image: "Image") -> PotentialEnergy:
        r"""
        Value of the IDPP objective function for a single image defined by,

        .. math::

            S_k = 0.5 Σ_i  Σ_{j \ne i} w(r_{ij}) (r_{ij}^{(k)} - r_{ij})^2

        where :math:`i` and :math:`j` enumerate over atoms for an image indexed
        by :math:`k`.

        -----------------------------------------------------------------------
        Arguments:
            image: NEB image (k)

        Returns:
            (float): :math:`S_k`
        """
        r_k, r = self._req_distance_matrix(image), self._distance_matrix(image)
        w = self._weight_matrix(image)

        return PotentialEnergy(0.5 * np.sum(w * (r_k - r) ** 2))

    def grad(self, image: "Image") -> np.ndarray:
        r"""
        Gradient of the potential with respect to displacement of
        the Cartesian components: :math:`\nabla S = (dS/dx_0, dS/dy_0, dS/dz_0,
        dS/dx_1, ...)` where the numbers denote different atoms. For example,

        .. math::

            \frac{dS}{dx_0} = -2 \sum_{i \ne j}
                               \left[2(c-r_{ij})r_{ij}^{-6}
                                     + w(r_{ij})r_{ij}^{-1})
                               \right](c - r_{ij})(x_0 - x_j)

        where :math:`c = r_{ij}^{(k)}`.

        -----------------------------------------------------------------------
        Arguments:
            image: NEB image (k)

        Returns:
            (np.ndarray): :math:`\nabla S`
        """

        x = np.array(image.coordinates).flatten()
        grad = np.zeros_like(x)

        r = self._distance_matrix(image, unity_diagonal=True)
        w = self._weight_matrix(image)
        r_k = self._req_distance_matrix(image)

        a = -2 * (2 * (r_k - r) ** 2 * r ** (-6) + w * (r_k - r) * r ** (-1))

        """
        The following numpy operations are the same as:
        -----------------------------------------------------------------------
        x = x.reshape((-1, 3))
        grad = np.zeros_like(x)

        for i in range(n_atoms):
            for j in range(n_atoms):

                if i != j:
                    grad[i, :] += a[i, j] * (x[i, :] - x[j, :])
        -----------------------------------------------------------------------
        """

        a[self._diagonal_distance_matrix_idxs] = 0.0
        delta = np.subtract.outer(x, x)

        grad[0::3] = np.sum(a * delta[0::3, 0::3], axis=1)  # x
        grad[1::3] = np.sum(a * delta[1::3, 1::3], axis=1)  # y
        grad[2::3] = np.sum(a * delta[2::3, 2::3], axis=1)  # z

        return grad.reshape((-1, 3))

    def _set_distance_matrices(self, images: "Images") -> None:
        """
        For each image determine the optimum distance matrix using

        .. math::

            r_{ij}^{(k)} = r_{ij}^{(1)} + k (r_{ij}^{(N)} - r_{ij}^{(1)}) / N

        and set the the diagonal indices of each distance matrix.
        """

        dist_mat_1 = self._distance_matrix(image=images[0])
        dist_mat_n = self._distance_matrix(image=images[-1])

        delta = dist_mat_n - dist_mat_1
        n = len(images)

        for k, image in enumerate(images):
            self._dists[image.name] = dist_mat_1 + k * delta / n

        self._diagonal_distance_matrix_idxs = np.diag_indices_from(delta)
        return None

    def _req_distance_matrix(self, image: "Image"):
        """Required distance matrix for an image, with elements r_{ij}^k"""
        return self._dists[image.name]

    def _distance_matrix(
        self, image: "Image", unity_diagonal: bool = False
    ) -> np.ndarray:
        """Distance matrix for an image"""

        x = image.coordinates
        r = distance_matrix(x, x)

        if unity_diagonal:
            r[self._diagonal_distance_matrix_idxs] = 1.0

        return r

    def _weight_matrix(self, image: "Image") -> np.ndarray:
        r"""
        Weight matrix with elements

        .. math::

            w_{ij} = 1/r_{ij}^4


        for :math:`i \ne j` otherwise :math:`w_{ii} = 0`
        """
        r = self._distance_matrix(image, unity_diagonal=True)
        w = r ** (-4.0)
        w[self._diagonal_distance_matrix_idxs] = 0.0  # Zero w_ii elements

        return w
