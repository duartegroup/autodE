"""
Tests for the physical constants (CODATA 2022) and corresponding unit conversions.
"""

import numpy as np
import pytest

from autode.values import Energy, Gradient
from autode import Molecule
from autode.units import (
    ev,
    kjmol,
    kcalmol,
    J,
    a0,
    deg,
    ha_per_ang,
    ha_per_a0,
    ev_per_ang,
    kcalmol_per_ang,
)


@pytest.fixture
def water():
    water = Molecule(name="h2o", smiles="O")
    water.coordinates = np.array(
        [
            [0.00080, -0.01527, 0.00000],
            [-0.77064, 0.55529, 0.00000],
            [0.77315, 0.55408, 0.00000],
        ]
    )
    water.energy = -5.070544093022
    water.gradient = Gradient(
        np.array(
            [
                [
                    -9.8758339297319e-06,
                    -4.8207524887615e-04,
                    7.3938043560707e-17,
                ],
                [
                    -1.0298858507513e-04,
                    2.3761821380166e-04,
                    -9.9428684281661e-17,
                ],
                [
                    1.1286441900486e-04,
                    2.4445703507450e-04,
                    2.5490640720954e-17,
                ],
            ]
        ),
        units=ha_per_a0,
    )

    return water


@pytest.mark.parametrize(
    "units, expected",
    [
        (ev, -137.97653379249905),
        (kjmol, -13312.71168819246),
        (kcalmol, -3181.8144570249665),
        (J, -2.2106277848265473e-17),
    ],
)
def test_h2o_xtb_energy_conversion(water, units, expected):
    converted = water.energy.to(units)

    assert converted.units == units
    assert np.isclose(float(converted), expected, rtol=1e-12, atol=0)


def test_energy_conversion_between_non_base_units():
    energy = Energy(-137.97653379249905, units=ev)

    assert np.isclose(
        float(energy.to(kcalmol)), -3181.8144570249665, rtol=1e-12, atol=0
    )
    assert np.isclose(
        float(energy.to(kjmol)), -13312.71168819246, rtol=1e-12, atol=0
    )


@pytest.mark.parametrize(
    "units, expected",
    [
        (ha_per_ang, -0.0009109901924547569),
        (ev_per_ang, -0.024789305993186954),
        (kcalmol_per_ang, -0.5716549765437422),
    ],
)
def test_h2o_xtb_gradient_conversion(water, units, expected):
    converted = water.gradient.to(units)

    assert converted.units == units
    # y component of the gradient on the oxygen atom
    assert np.isclose(converted[0, 1], expected, rtol=1e-12, atol=0)


def test_length_conversion(water):
    r_oh = water.distance(0, 1)

    assert np.isclose(
        float(r_oh.to(a0)), 1.813210078242017, rtol=1e-10, atol=0
    )


def test_angle_conversion(water):
    # H-O-H angle
    theta = water.angle(1, 0, 2)

    assert np.isclose(
        float(theta.to(deg)), 107.11685792545026, rtol=1e-10, atol=0
    )
