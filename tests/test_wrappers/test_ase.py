import sys

import numpy as np
import pytest

from ase.calculators.calculator import Calculator, all_changes
from ase.calculators.emt import EMT

from autode.atoms import Atom
from autode.calculations import Calculation
from autode.calculations.types import CalculationType
from autode.exceptions import CalculationException, CouldNotGetProperty
from autode.species.molecule import Molecule
from autode.wrappers.ASE import ASE
from autode.wrappers.keywords import GradientKeywords, SinglePointKeywords

# Using EMT that is shipped with ase as a deterministic, dependency-free
# calculator to test this wrapper with
method = ASE(calculator_factory=EMT)


# EMT only parameterises a handful of metals (Al, Ni, Cu, Pd, Ag, Pt, Au),
# so Cu is used as the fixture element
def cu2(charge: int = 0, mult: int = 1, x: float = 2.5) -> Molecule:
    return Molecule(
        name="Cu2",
        atoms=[Atom("Cu"), Atom("Cu", x=x)],
        charge=charge,
        mult=mult,
    )


class EnergyOnlyCalculator(Calculator):
    """An ASE calculator that implements energy but not forces, used to
    exercise the 'gradient genuinely unavailable' code path"""

    implemented_properties = ["energy"]

    def calculate(
        self, atoms=None, properties=None, system_changes=all_changes
    ):
        super().calculate(atoms, properties, system_changes)
        self.results["energy"] = 0.0


class FailingCalculator(Calculator):
    """An ASE calculator whose energy evaluation always raises"""

    implemented_properties = ["energy", "forces"]

    def calculate(
        self, atoms=None, properties=None, system_changes=all_changes
    ):
        raise RuntimeError("Calculator is broken")


def test_repr():
    assert "ase" in repr(method).lower()


def test_uses_external_io_is_false():
    assert method.uses_external_io is False


def test_implements():
    assert method.implements(CalculationType.energy)
    assert method.implements(CalculationType.gradient)
    assert not method.implements(CalculationType.opt)
    assert not method.implements(CalculationType.hessian)


def test_is_available(monkeypatch):
    assert method.is_available

    monkeypatch.setitem(sys.modules, "ase", None)
    assert not method.is_available


def test_ase_atoms_from_species():
    mol = cu2(charge=-1, mult=2)

    atoms = ASE._ase_atoms_from(mol)

    assert atoms.get_chemical_symbols() == ["Cu", "Cu"]
    assert np.allclose(atoms.get_positions(), mol.coordinates)
    assert atoms.info["charge"] == -1
    assert atoms.info["spin"] == 2


def test_single_point_sets_energy_not_gradient():
    mol = cu2()

    calc = Calculation(
        name="sp",
        molecule=mol,
        method=method,
        keywords=SinglePointKeywords(),
    )
    calc.run()

    assert mol.energy is not None
    assert mol.gradient is None


def test_gradient_calculation():
    mol = cu2()

    calc = Calculation(
        name="grad",
        molecule=mol,
        method=method,
        keywords=GradientKeywords(),
    )
    calc.run()

    assert mol.energy is not None
    assert mol.gradient is not None
    assert mol.gradient.shape == (2, 3)


def test_gradient_matches_finite_difference():
    delta_r = 1e-5

    mol = cu2()
    mol_disp = cu2(x=2.5 + delta_r)

    mol.single_point(method)
    mol_disp.single_point(method)

    delta_energy = mol_disp.energy - mol.energy
    fd_grad = delta_energy / delta_r

    calc = Calculation(
        name="grad",
        molecule=mol,
        method=method,
        keywords=GradientKeywords(),
    )
    calc.run()

    assert np.isclose(mol.gradient[1, 0], fd_grad, atol=1e-4)


def test_energy_evaluation_failure_raises():
    mol = cu2()
    failing_method = ASE(calculator_factory=FailingCalculator)

    calc = Calculation(
        name="sp",
        molecule=mol,
        method=failing_method,
        keywords=SinglePointKeywords(),
    )

    with pytest.raises(CalculationException):
        calc.run()


def test_requested_gradient_unavailable_raises():
    mol = cu2()
    forceless_method = ASE(calculator_factory=EnergyOnlyCalculator)

    calc = Calculation(
        name="grad",
        molecule=mol,
        method=forceless_method,
        keywords=GradientKeywords(),
    )

    with pytest.raises(CouldNotGetProperty):
        calc.run()
