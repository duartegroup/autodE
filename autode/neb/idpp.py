import numpy as np


class IDPP:
    """
    Image dependent pair potential (IDPP) objective function from
    https://arxiv.org/pdf/1406.1512.pdf

    .. math::

        S = Σ  Σ  w(r_{ij}) (r_{ij}^k - r_{ij})^2
           i j>i

    where :math:`r_{ij}` is the distance between atoms i and j and
    :math:`r_{ij}^k = r_{ij}^0 + (r_{ij}^N - r_{ij}^0)/N` for :math:`N` images.
    Note: here the superscripts denote the image number, not a power. The
    weight function is :math:`w(r_ij) = r_{ij}^{-4}`, as suggested in the paper
    """

    def __init__(self, images: 'autode.neb.original.Images'):
        """Initialise a IDPP potential"""

    def __call__(self, image_idx: int) -> float:
        """Value"""

    def grad(self, image_idx: int) -> np.ndarray:
        """Gradient of the potential"""
