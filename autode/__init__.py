import importlib.metadata

from autode import methods
from autode import geom
from autode import pes
from autode import utils
from autode import neb
from autode import mol_graphs
from autode import hessians
from autode.neb import NEB, CINEB
from autode.reactions.reaction import Reaction
from autode.reactions.multistep import MultiStepReaction
from autode.transition_states.transition_state import TransitionState
from autode.atoms import Atom
from autode.species.molecule import Reactant, Product, Molecule, Species
from autode.species.complex import NCIComplex
from autode.config import Config
from autode.calculations import Calculation
from autode.wrappers.keywords import (
    KeywordsSet,
    OptKeywords,
    HessianKeywords,
    SinglePointKeywords,
    Keywords,
    GradientKeywords,
)
from autode.utils import temporary_config

"""
Bumping the version number requires following the release procedure:

- Run tests/benchmark.py with both organic and organometallic sets

- Release on conda-forge
  - Fork https://github.com/conda-forge/autode-feedstock
  - Make a local branch
  - Modify recipe/meta.yaml with the new version number, sha256
  - Push commit and open PR on the conda-forge feedstock
  - Merge when tests pass
"""

__version__ = importlib.metadata.version("autode")

__all__ = [
    "KeywordsSet",
    "Keywords",
    "OptKeywords",
    "HessianKeywords",
    "SinglePointKeywords",
    "GradientKeywords",
    "Reaction",
    "MultiStepReaction",
    "Atom",
    "Species",
    "Reactant",
    "Product",
    "Molecule",
    "TransitionState",
    "NCIComplex",
    "Config",
    "Calculation",
    "NEB",
    "CINEB",
    "pes",
    "neb",
    "geom",
    "methods",
    "mol_graphs",
    "utils",
    "hessians",
]
