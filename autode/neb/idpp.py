import numpy as np
from scipy.spatial import distance_matrix


class IDPP:
    """
    Image dependent pair potential (IDPP) objective function from
    https://arxiv.org/pdf/1406.1512.pdf

    .. math::

        S = Σ  Σ  w(r_{ij}) (r_{ij}^k - r_{ij})^2
           i j>i

    where :math:`r_{ij}` is the distance between atoms i and j and
    :math:`r_{ij}^k = r_{ij}^0 + k(r_{ij}^N - r_{ij}^0)/N` for :math:`N` images.
    Note: here the superscripts denote the image number, not a power. The
    weight function is :math:`w(r_ij) = r_{ij}^{-4}`, as suggested in the paper
    """

    def __init__(self, images: 'autode.neb.original.Images'):
        """Initialise a IDPP potential"""

        if len(images) < 2:
            raise ValueError('Must have at least 2 images for IDPP')

        # Distance matrices containing all r_{ij}^k
        self._dists = {image.name: None for k, image in enumerate(images)}
        self._diagonal_distance_matrix_idxs = None

        self._set_distance_matrices(images)

    def __call__(self,
                 image: 'autode.neb.original.Image'
                 ) -> float:
        """Value of the IDPP objective function for a single image


        S = 0.5 Σ  Σ  w(r_{ij}) (r_{ij}^k - r_{ij})^2
                i j≠i
        """
        r_k, r = self._req_distance_matrix(image), self._distance_matrix(image)

        return 0.5 * np.sum(self._w(image) * (r_k - r)**2)

    def grad(self,
             image: 'autode.neb.original.Image'
             ) -> np.ndarray:
        """
        Gradient of the potential with respect to displacement of
        the Cartesian components: (dS/dx0, dS/dy0, dS/dz0, dS/dx1, ...)
        where the numbers denote different atoms
        """

        n_atoms = image.species.n_atoms
        x = image.species.coordinates
        grad = np.zeros(shape=(n_atoms, 3))

        r = self._distance_matrix(image, unity_diagonal=True)
        w = self._w(image)
        r_k = self._req_distance_matrix(image)

        a = -2 * (2 * (r_k - r)**2 * r**(-6)
                  + w * (r_k - r) * r**(-1))

        for i in range(n_atoms):
            for j in range(n_atoms):

                if i == j:
                    continue

                grad[i, :] += a[i, j] * (x[i, :] - x[j, :])

        return grad.flatten()

    def _set_distance_matrices(self, images) -> None:
        """
        For each image determine the optimum distance matrix using

        .. math::

            r_{ij}^k = r_{ij}^0 + k (r_{ij}^N - r_{ij}^0) / N

        and set the identity distance matrix e.g. I_n for an image with n atoms
        """

        dist_mat1 = self._distance_matrix(image=images[0])
        dist_matN = self._distance_matrix(image=images[-1])

        delta = (dist_matN - dist_mat1)
        n = len(images)

        for k, image in enumerate(images):
            self._dists[image.name] = dist_mat1 + k * delta / n

        self._diagonal_distance_matrix_idxs = np.diag_indices_from(delta)
        return None

    def _req_distance_matrix(self, image):
        """Required distance matrix for an image, with elements r_{ij}^k"""
        return self._dists[image.name]

    def _distance_matrix(self, image, unity_diagonal=False) -> np.ndarray:
        """Distance matrix for an image"""

        x = image.species.coordinates
        r = distance_matrix(x, x)

        if unity_diagonal:
            r[self._diagonal_distance_matrix_idxs] = 1.0

        return r

    def _w(self, image) -> np.ndarray:
        """
        Weight matrix with elements:

        .. math::

            w_ij = 1/r_ij^4

        for :math:`i \ne j` otherwise :math:`w_ii = 0`
        """
        w = self._distance_matrix(image)

        w[self._diagonal_distance_matrix_idxs] = 1.0  # Don't divide by zero
        w = w**(-4.0)
        w[self._diagonal_distance_matrix_idxs] = 0.0  # Zero w_ii elements

        return w
