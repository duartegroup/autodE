from abc import ABC, abstractmethod
from autode.opt.optimisers.base import NullOptimiser


class Method(ABC):

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Name of this method. Should be a non unique identifier for all methods
        of this class. e.g. "g09" for Gaussian 09
        """

    @property
    def optimiser(self) -> "autode.opt.optimisers.base.BaseOptimiser":
        """"""
        return NullOptimiser()
