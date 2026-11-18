import numpy as np
import autode.wrappers.methods

from typing import Callable, List, Optional, TYPE_CHECKING

from autode.calculations.types import CalculationType as ct
from autode.exceptions import CalculationException, CouldNotGetProperty
from autode.values import Gradient, PotentialEnergy
from autode.wrappers.keywords import GradientKeywords, KeywordsSet

if TYPE_CHECKING:
    from ase import Atoms as ASEAtoms
    from ase.calculators.calculator import Calculator as ASECalculator
    from autode.calculations.executors import CalculationExecutor
    from autode.calculations.types import CalculationType
    from autode.species.species import Species


class ASE(autode.wrappers.methods.Method):
    """
    An autodE method that evaluates energies/gradients using an ASE
    Calculator. Example::

        >>> from mace.calculators import mace_mp
        >>> from autode.wrappers.ASE import ASE
        >>> method = ASE(calculator_factory=mace_mp)

    Only energies and gradients are implemented; optimisations and Hessians
    fall back to autodE's own internal optimisers and numerical Hessian
    evaluation respectively, both of which are driven by repeated gradient
    evaluations through this method.
    """

    def __init__(
        self,
        calculator_factory: Callable[[], "ASECalculator"],
        name: str = "ase",
        doi_list: Optional[List[str]] = None,
    ):
        """
        -----------------------------------------------------------------------
        Arguments:
            calculator_factory: A zero-argument callable returning a fresh
                                 ASE Calculator instance. A factory, rather
                                 than a calculator instance, is required
                                 because autodE may evaluate gradients in
                                 separate worker processes (e.g. for a
                                 numerical Hessian) and many ASE calculators
                                 (e.g. ones backed by a torch model) are not
                                 safe to share across processes.

            name: Name of this method

            doi_list: Digital object identifiers for the underlying
                      potential, for citation purposes
        """
        super().__init__(
            name=name,
            keywords_set=KeywordsSet(),
            doi_list=[] if doi_list is None else doi_list,
        )
        self._calculator_factory = calculator_factory

    def __repr__(self):
        return f"ASE(available = {self.is_available})"

    @property
    def uses_external_io(self) -> bool:
        return False

    def implements(self, calculation_type: "CalculationType") -> bool:
        return calculation_type in (ct.energy, ct.gradient)

    @property
    def is_available(self) -> bool:
        try:
            import ase  # noqa: F401
        except ImportError:
            return False

        return True

    @staticmethod
    def _ase_atoms_from(species: "Species") -> "ASEAtoms":
        """
        Construct an ase.Atoms from an autodE Species/Molecule. Charge and
        spin multiplicity are set in Atoms.info, following the convention
        used by charge/spin-aware foundation models (e.g. MACE-OMol, UMA)
        to read them; calculators that don't use these keys simply ignore
        them
        """
        from ase import Atoms as ASEAtoms

        atoms = ASEAtoms(
            symbols=[atom.label for atom in species.atoms],
            positions=np.array(species.coordinates),
        )
        atoms.info["charge"] = species.charge
        atoms.info["spin"] = species.mult

        return atoms

    def execute(self, calc: "CalculationExecutor") -> None:
        """
        Evaluate the energy, and where possible the gradient, of
        calc.molecule directly using the wrapped ASE calculator, setting
        the properties on calc.molecule in place
        """
        atoms = self._ase_atoms_from(calc.molecule)
        atoms.calc = self._calculator_factory()

        try:
            energy = atoms.get_potential_energy()
        except Exception as exc:
            raise CalculationException(
                f"Failed to evaluate the energy of {calc.molecule} with "
                f"{self}"
            ) from exc

        calc.molecule.energy = PotentialEnergy(
            energy, units="eV", method=self, keywords=calc.input.keywords
        )

        if isinstance(calc.input.keywords, GradientKeywords):
            try:
                forces = np.array(atoms.get_forces())
            except Exception as exc:
                raise CouldNotGetProperty(name="gradient") from exc

            calc.molecule.gradient = Gradient(-forces, units="eV Å^-1")

        return None

    def terminated_normally_in(self, calc: "CalculationExecutor") -> bool:
        return calc.molecule.energy is not None
