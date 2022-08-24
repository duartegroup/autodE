from enum import Enum


class CalculationType(Enum):
    """Enum defining a mode of a calculation"""

    opt = 0
    energy = 1
    gradient = 2
    hessian = 3
