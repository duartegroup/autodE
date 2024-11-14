# distutils: language = c++
# distutils: sources = [autode/ext/src/idpp.cpp, autode/ext/src/utils.cpp]

import numpy as np
from autode.ext.wrappers cimport calculate_idpp_path

class IDPP:
    def __init__(
        self,
        init_coords,
        final_coords, n_images, k_spr = 0.1, sequential = True
    ):
        """


        Args:
            init_coords:
            final_coords:
            n_images:
            k_spr:
            sequential:
        """
        self.init_coords = init_coords
        self.final_coords = final_coords
        assert (n_images > 2), "Must have more than 2 images"
        self.n_images = n_images
        assert (k_spr > 0), "Spring constant must be positive"
        self.k_spr = k_spr