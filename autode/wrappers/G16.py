from autode.wrappers.G09 import G09
from autode.config import Config


class G16(G09):
    """Gaussian 16 seems to have the same syntax as Gaussian 09"""

    def __init__(self):
        super().__init__(
            executable_name="g16",
            path=Config.G16.path,
            keywords_set=Config.G16.keywords,
            implicit_solvation_type=Config.G16.implicit_solvation_type,
        )

    def __repr__(self):
        return f"Gaussian16(available = {self.is_available})"


g16 = G16()
