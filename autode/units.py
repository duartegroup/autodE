from typing import Union, Collection, Optional
from autode.constants import Constants


class Unit:
    def __str__(self):
        return f"Unit({self.name})"

    def __repr__(self):
        return self.__str__()

    def lower(self):
        """Lower case name of the unit"""
        return self.name.lower()

    def __eq__(self, other):
        """Equality of two units"""
        return other.lower() in self.aliases

    def __init__(
        self,
        name: str,
        conversion: float,
        aliases: Optional[Collection] = None,
        plot_name: Optional[str] = None,
    ):
        """
        Unit

        ----------------------------------------------------------------------
        Arguments:
            name (str):

            conversion (float): Conversion from default units to the new

        Keyword Arguments:
            aliases (list | set | tuple | None): Set of name aliases for this
                                                 unit

            plot_name (str | None): Name to use if this unit is used in a plot
        """

        self.name = name
        self.conversion = conversion

        self.aliases = [name.lower()]
        if aliases is not None:
            self.aliases += [alias.lower() for alias in aliases]

        self.plot_name = plot_name if plot_name is not None else name


class BaseUnit(Unit):
    """A unit in the base unit system, thus an identity conversion factor"""

    def __init__(
        self,
        name: str,
        aliases: Union[Collection, None] = None,
        plot_name: Union[str, None] = None,
    ):

        super().__init__(
            name, conversion=1.0, aliases=aliases, plot_name=plot_name
        )


class CompositeUnit(Unit):
    def __init__(
        self,
        *args: Unit,
        per: Collection[Unit] = None,
        name: Union[str, None] = None,
        aliases: Union[Collection, None] = None,
    ):
        """
        A unit as a composite of others, e.g. Ha Å^-1

        Arguments:
            args (autode.units.Unit): Units on the numerator

            per (list(autode.units.Unit) | None): Units on the denominator
        """
        per = [] if per is None else per

        if name is None:
            top_names = " ".join([u.name for u in args])
            per_names = " ".join([u.name for u in per])
            name = f"{top_names}({per_names})^-1"

        conversion = 1.0
        for unit in args:
            conversion *= unit.conversion

        for unit in per:
            conversion /= unit.conversion

        super().__init__(name=name, conversion=conversion, aliases=aliases)


# ----------------------------- Energies -------------------------------


ha = BaseUnit(name="Ha", aliases=["hartree", "Eh"], plot_name="Ha")


ev = Unit(
    name="eV",
    conversion=Constants.ha_to_eV,
    aliases=["electron volt", "electronvolt"],
    plot_name="eV",
)


# Upper case name to maintain backwards compatibility
kjmol = KjMol = Unit(
    name="kJ mol-1",
    conversion=Constants.ha_to_kJmol,
    aliases=["kjmol", "kjmol-1", "kj mol^-1", "kj", "kj mol"],
    plot_name="kJ mol$^{-1}$",
)


kcalmol = KcalMol = Unit(
    name="kcal mol-1",
    conversion=Constants.ha_to_kcalmol,
    aliases=["kcalmol", "kcalmol-1", "kcal mol^-1", "kcal", "kcal mol"],
    plot_name="kcal mol$^{-1}$",
)

J = Unit(name="J", conversion=Constants.ha_to_J, aliases=["joule"])


def energy_unit_from_name(name: str) -> "autode.units.Unit":
    """
    Generate an energy unit given a name

    ---------------------------------------------------------------------------
    Arguments:
        name: Name of the unit

    Raises:
        (StopIteration): If a suitable energy unit is not found
    """

    for unit in (ha, ev, kcalmol, kjmol, J):
        if name.lower() in unit.aliases:
            return unit

    raise StopIteration(
        f"Failed to convert {name} to a valid energy unit "
        f"must be one of: {ha, ev, kcalmol, kjmol, J}"
    )


# ----------------------------------------------------------------------
# ------------------------------ Angles --------------------------------

rad = BaseUnit(name="rad", aliases=["radians", "rads", "radian"])


deg = Unit(
    name="°", conversion=Constants.rad_to_deg, aliases=["deg", "degrees", "º"]
)

# ----------------------------------------------------------------------
# ---------------------------- Distances -------------------------------

ang = BaseUnit(name="Å", aliases=["ang", "angstrom"])


a0 = Unit(name="bohr", conversion=Constants.ang_to_a0, aliases=["a0"])

nm = Unit(
    name="nm",
    conversion=Constants.ang_to_nm,
    aliases=["nanometer", "nano meter"],
)

pm = Unit(
    name="pm",
    conversion=Constants.ang_to_pm,
    aliases=["picometer", "pico meter"],
)

m = Unit(name="m", conversion=Constants.ang_to_m, aliases=["meter"])


ang_amu_half = BaseUnit(
    name="Å amu^1/2", aliases=["ang amu^1/2", "Å amu^0.5", "ang amu^0.5"]
)

# ----------------------------------------------------------------------
# ------------------------------ Masses --------------------------------

amu = BaseUnit(name="amu", aliases=["Da", "g mol-1", "g mol^-1", "g/mol"])

kg = Unit(name="kg", conversion=Constants.amu_to_kg)

m_e = Unit(name="m_e", conversion=Constants.amu_to_me, aliases=["me"])

# ----------------------------------------------------------------------
# -------------------- Mass-weighted distance squared ------------------

amu_ang_sq = CompositeUnit(amu, ang, ang, name="amu Å^2")

kg_m_sq = CompositeUnit(kg, m, m, name="kg m^2")

# ----------------------------------------------------------------------
# ----------------------------- Gradients ------------------------------


ha_per_ang = CompositeUnit(
    ha, per=[ang], aliases=["ha / Å", "ha Å-1", "ha Å^-1", "ha/ang"]
)

ha_per_a0 = CompositeUnit(
    ha, per=[a0], aliases=["ha / a0", "ha a0-1", "ha a0^-1", "ha/bohr"]
)

ev_per_ang = CompositeUnit(
    ev, per=[ang], aliases=["ev / Å", "ev Å^-1", "ev/ang"]
)

kcalmol_per_ang = CompositeUnit(
    kcalmol,
    per=[ang],
    aliases=["ha kcal mol-1", "ha/kcal mol-1", "kcal mol^-1 Å^-1"],
)

# ----------------------------------------------------------------------
# ------------------------- 2nd derivatives ----------------------------

ha_per_ang_sq = CompositeUnit(
    ha,
    per=[ang, ang],
    name="Ha Å^-2",
    aliases=["Ha / Å^2", "ha/ang^2", "ha/ang2", "ha ang^-2"],
)

ha_per_a0_sq = CompositeUnit(
    ha,
    per=[a0, a0],
    name="Ha a0^-2",
    aliases=[
        "ha/bohr^2",
        "ha/bohr2",
        "ha bohr^-2",
        "ha/a0^2",
        "ha/a02",
        "ha a0^-2",
    ],
)

J_per_ang_sq = CompositeUnit(
    J, per=[ang, ang], name="J ang^-2", aliases=["J/ang^2", "J/ang2", "J ang2"]
)

J_per_m_sq = CompositeUnit(
    J, per=[m, m], name="J m^-2", aliases=["J/m^2", "J/m2", "J m2"]
)

J_per_ang_sq_kg = CompositeUnit(J, per=[ang, ang, kg], name="J m^-2 kg^-1")


# ----------------------------------------------------------------------
# --------------------------- Frequencies ------------------------------

wavenumber = BaseUnit(name="cm^-1", aliases=["cm-1", "per cm", "/cm"])

hz = Unit(
    name="s^-1", conversion=Constants.per_cm_to_hz, aliases=["hz", "s-1", "/s"]
)


# ----------------------------------------------------------------------
# ------------------- Digital storage allocation -----------------------


byte = Unit(
    name="byte", conversion=1e6, aliases=["bytes"]
)  # 1,000,000 bytes = 1 MB

MB = BaseUnit(name="mb", aliases=["megabyte"])

GB = Unit(name="gb", conversion=1e-3, aliases=["gigabyte"])  # 1000 MB = 1 GB

TB = Unit(name="tb", conversion=1e-6, aliases=["terabyte"])  # 1000 GB = 1 TB
