"""
Thermochemistry calculation from frequencies and coordinates. Copied from
otherm (https://github.com/duartegroup/otherm) 16/05/2021. See
autode/common/thermochemistry.pdf for mathematical background


All calculations performed in SI units, for simplicity.
"""
import numpy as np
from enum import Enum
from typing import TYPE_CHECKING, Any
from autode.log import logger
from autode.config import Config
from autode.constants import Constants
from autode.values import FreeEnergyCont, EnthalpyCont, Frequency, Temperature

if TYPE_CHECKING:
    from autode.species.species import Species


class SIConstants:
    """Constants in SI (International System of Units) units"""

    k_b = 1.38064852e-23  # J K-1
    h = 6.62607004e-34  # J s
    c = 299792458  # m s-1


class LFMethod(Enum):
    """Method to treat low-frequency modes

    See Also:
        autode.thermochemistry.igm.calculate_thermo_cont for citations for the
        different methods.
    """

    igm = 0
    truhlar = 1
    grimme = 2
    minenkov = 3


class _ThermoParams:
    def __init__(self, default_sigma_r: int = 1, **kwargs: Any) -> None:
        self.method = kwargs.get("lfm_method", LFMethod[Config.lfm_method])
        if isinstance(self.method, str):
            self.method = LFMethod[self.method.lower()]

        self.T: float = kwargs["T"]
        self.ss = kwargs.get("ss", Config.standard_state)
        self.shift = Frequency(kwargs.get("freq_shift", Config.vib_freq_shift))
        self.w0 = Frequency(kwargs.get("w0", Config.grimme_w0))
        self.alpha = int(kwargs.get("alpha", Config.grimme_alpha))
        self.sigma_r = kwargs.get("sn", default_sigma_r)


def calculate_thermo_cont(
    species: "Species",
    temp: Temperature = Temperature(298.15, units="K"),
    **kwargs: Any,
):
    """
    Calculate and set the thermochemical contributions (Enthalpic, Free Energy)
    of a species using a variant of the ideal gas model (RRHO). Appends
    energies to species.energies. Default methods are set in
    autode.config.Config. See references:

    [1] Chem. Eur. J. 2012, 18, 9955

    [2] J. Phys. Chem. B, 2011, 115, 14556

    [3] J. Comput. Chem., 2023, 44, 1807

    ---------------------------------------------------------------------------
    Arguments:
        species (autode.species.Species):

        temp (autode.values.Temperature): Temperature in K. Default: 298.15 K

    Keyword Arguments:
        lfm_method (str | LFMethod): Method used to calculate the molecular
                                    entropy by treating the low frequency modes.
                                    One of: {'igm', 'truhlar', 'grimme', 'minenkov'}.
                                    Default: Config.lfm_method

        ss (str): Standard state at which the molecular entropy is calculated.
                  Should be 1M for a solution phase molecule and 1 atm for
                  a molecule in the gas phase. Default: Config.standard_state

        shift (float | Frequency): Frequency parameters for Truhlar's method,
                                   if float then assumed cm-1 units. Only used
                                   if method='truhlar'.
                                   Default: Config.vib_freq_shift

        w0 (float | Frequency): ω0 parameter, if float then assumed cm-1 units
                                Default: Config.grimme_w0

        alpha (int | float): α parameter in Grimme's qRRHO method.
                             Default: Config.grimme_w0

        sn (int): The symmetry number, if not present then will default to
                  species.sn

    Raises:
        (KeyError | ValueError): If frequencies are not defined
    """
    params = _ThermoParams(
        default_sigma_r=species.sn,
        T=float(temp.to("K")) if isinstance(temp, Temperature) else temp,
        **kwargs,
    )

    if species.n_atoms == 0:
        logger.warning(
            "Species had no atoms. Cannot calculate thermochemical "
            "contributions (G_cont, H_cont)"
        )
        return

    if species.frequencies is None and species.n_atoms > 1:
        raise ValueError(
            "Cannot calculate vibrational entropy/internal energy"
            f" no frequencies present for {species.name}."
        )

    logger.info(
        f"Calculating themochemistry with {params.method} at {params.T} K"
    )

    S = _entropy(species, params)
    U = _internal_energy(species, params)
    H = EnthalpyCont(U + SIConstants.k_b * params.T, units="J").to("Ha")

    # Add a method string for how this enthalpic contribution was calculated
    H.method_str = _thermo_method_str(species, **kwargs)
    species.energies.append(H)

    G = FreeEnergyCont(H.to("J") - params.T * S, units="J").to("Ha")

    # Method used to calculate the free energy is the  same as the enthalpy..
    G.method_str = H.method_str
    species.energies.append(G)

    return None


def _thermo_method_str(species: "Species", **kwargs: Any) -> str:
    """
    Brief summary of the important methods used in evaluating the free energy
    using entropy and enthalpy methods

    ---------------------------------------------------------------------------
    Arguments:
        species (autode.species.Species):

        **kwargs:

    Returns:
        (str):
    """
    string = ""

    if species.energy is not None:
        string += f"{species.energy.method_str} "

    string += (
        f'{kwargs.get("ss", Config.standard_state)} standard state, '
        f'using a {kwargs.get("lfm_method", Config.lfm_method)} '
        f"treatment of low-frequency modes to the entropy."
    )

    return string


def _q_trans_igm(species: "Species", ss: str, temp: float) -> float:
    """
    Calculate the translational partition function using the PIB model,
    coupled with an effective volume

    ---------------------------------------------------------------------------
    Arguments:
        species (autode.species.Species):
        ss (str): Standard state to use. One of: {1M, 1atm}
        temp (float): Temperature in K

    Returns:
        (float): Translational partition function q_trns
    """

    if ss.lower() == "1atm":
        effective_volume = SIConstants.k_b * temp / Constants.atm_to_pa

    elif ss.lower() == "1m":
        effective_volume = 1.0 / (
            Constants.n_a * (1.0 / Constants.dm_to_m) ** 3
        )

    else:
        raise ValueError(
            f"Cannot calculate PIB partition function using a"
            f' {ss} state. Only "1atm" and "1m" implemented'
        )

    q_trans = (
        2.0
        * np.pi
        * species.weight.to("kg")
        * SIConstants.k_b
        * temp
        / SIConstants.h**2
    ) ** 1.5 * effective_volume

    return q_trans


def _q_rot_igm(species: "Species", temp: float, sigma_r: int) -> float:
    """
    Calculate the rotational partition function using the IGM method. Uses the
    rotational symmetry number, default = 1

    ---------------------------------------------------------------------------
    Arguments:
        species (autode.species.Species):
        temp (float): Temperature in K
        sigma_r (int): Symmetry number e.g. 2 for water

    Returns:
        (float): Rotational partition function q_rot
    """

    if species.n_atoms == 1:
        return 1

    assert species.com is not None, "Must have a COM"
    assert species.atoms is not None, "Must have atoms"
    assert species.moi is not None, "Must have a moment of inertia"

    if species.is_linear():
        com = species.com.to("m")
        i_val = sum(
            atom.mass.to("kg") * np.linalg.norm(atom.coord.to("m") - com) ** 2
            for atom in species.atoms
        )

        return (
            temp
            * 8
            * np.pi**2
            * SIConstants.k_b
            * i_val
            / (sigma_r * SIConstants.h**2)
        )

    # otherwise a polyatomic..
    i_mat = species.moi.to("kg m^2")
    omega_diag = SIConstants.h**2 / (
        8.0 * np.pi**2 * SIConstants.k_b * np.diagonal(i_mat)
    )

    return temp**1.5 / sigma_r * np.sqrt(np.pi / np.prod(omega_diag))


def _s_trans_pib(species: "Species", ss: str, temp: float) -> float:
    """
    Calculate the translational entropy using a particle in a box model

    ---------------------------------------------------------------------------
    Arguments:
        species (autode.species.Species):
        ss (str): Standard state to use. One of: {1M, 1atm}. For calculating
                  the effective box size in the q_trans calculation
        temp (float): Temperature in K

    Returns:
        (float): S_trans
    """

    q_trans = _q_trans_igm(species, ss=ss, temp=temp)
    return SIConstants.k_b * (np.log(q_trans) + 1.0 + 1.5)


def _s_rot_rr(species: "Species", temp: float, sigma_r: int) -> float:
    """
    Calculate the rigid rotor (RR) entropy

    ---------------------------------------------------------------------------
    Arguments:
        species (autode.species.Species):
        temp (float): Temperature in K

    Returns:
        (float): S_rot
    """

    if species.n_atoms == 1:
        return 0

    q_rot = _q_rot_igm(species, temp=temp, sigma_r=sigma_r)

    if species.is_linear():
        return SIConstants.k_b * (np.log(q_rot) + 1.0)

    else:
        return SIConstants.k_b * (np.log(q_rot) + 1.5)


def _igm_s_vib(species: "Species", temp: float) -> float:
    """
    Calculate the entropy of a molecule according to the Ideal Gas Model (IGM)
    RRHO method

    ---------------------------------------------------------------------------
    Arguments:
        species (autode.species.Species):
        temp (float): Temperature in K

    Returns:
        (float): S_vib
    """
    assert species.vib_frequencies is not None, "Must have frequenecies"
    s = 0.0

    for freq in species.vib_frequencies:
        x = freq.real.to("hz") * SIConstants.h / (SIConstants.k_b * temp)
        s += SIConstants.k_b * (
            (x / (np.exp(x) - 1.0)) - np.log(1.0 - np.exp(-x))
        )

    return float(s)


def _truhlar_s_vib(
    species: "Species", temp: float, shift_freq: Frequency
) -> float:
    """
    Calculate the entropy of a molecule according to the Truhlar's method of
    shifting low frequency modes

    ---------------------------------------------------------------------------
    Arguments:
        species (autode.species.Species):
        temp (float): Temperature in K
        shift_freq (float): Shift all frequencies to this value

    Returns:
        (float): S_vib in J K-1 mol-1
    """
    assert species.vib_frequencies is not None, "Must have frequenecies"
    s = 0

    shift_cm = float(shift_freq.to("cm-1"))

    for freq in species.vib_frequencies:
        # Threshold lower bound of the frequency
        freq_cm = max(float(freq.to("cm-1").real), shift_cm)

        x = freq_cm * Constants.c_in_cm * SIConstants.h / SIConstants.k_b
        s += SIConstants.k_b * (
            ((x / temp) / (np.exp(x / temp) - 1.0))
            - np.log(1.0 - np.exp(-x / temp))
        )

    return float(s)


def _grimme_w(omega_0: float, freq: float, alpha: int) -> float:
    assert (
        abs(freq) < 6000 and abs(omega_0) < 6000
    ), "Units may be wrong - expecing cm-1"
    return 1.0 / (1.0 + (omega_0 / freq) ** alpha)


def _grimme_s_vib(
    species: "Species", temp: float, omega_0: Frequency, alpha: int
) -> float:
    """
    Calculate the entropy according to Grimme's qRRHO method of RR-HO
    interpolation in Chem. Eur. J. 2012, 18, 9955

    ---------------------------------------------------------------------------
    Arguments:
        species (autode.species.Species):
        temp (float): Temperature in K
        omega_0 (float | Frequency): ω0 parameter (cm-1)
        alpha (float): α parameter

    Returns:
        (float): S_vib
    """
    assert species.vib_frequencies is not None, "Must have frequenecies"
    assert species.moi is not None, "Must have a moment of inertia"

    s = 0.0
    w0 = float(omega_0.to("cm-1")) if hasattr(omega_0, "to") else omega_0

    # Average I = (I_xx + I_yy + I_zz) / 3.0
    b_avg = np.trace(species.moi.to("kg m^2")) / 3.0

    for freq in species.vib_frequencies:
        omega = float(freq.real.to("hz"))

        mu = SIConstants.h / (8.0 * np.pi**2 * omega)
        mu_prime = (mu * b_avg) / (mu + b_avg)

        x = omega * SIConstants.h / (SIConstants.k_b * temp)
        s_v = SIConstants.k_b * (
            (x / (np.exp(x) - 1.0)) - np.log(1.0 - np.exp(-x))
        )

        factor = (
            8.0 * np.pi**3 * mu_prime * SIConstants.k_b * temp
        ) / SIConstants.h**2
        s_r = SIConstants.k_b * (0.5 + np.log(np.sqrt(factor)))

        w = _grimme_w(omega_0=w0, freq=freq, alpha=alpha)

        s += w * s_v + (1.0 - w) * s_r

    return float(s)


def _entropy(species: "Species", params: _ThermoParams) -> float:
    """
    Calculate the entropy

    ---------------------------------------------------------------------------
    Arguments:
        species (autode.species.Species):
        params (autode.thermochemistry.igm._ThermoParams):

    Returns:
        (float): S in SI units

    Raises:
        (NotImplementedError):
    """
    logger.info(f"Calculating molecular entropy. σ_R = {params.sigma_r}")
    temp = params.T

    # Translational entropy component
    s_trans = _s_trans_pib(species, ss=params.ss, temp=params.T)

    if species.n_atoms < 2:
        # A molecule one or no atoms has no rotational/vibrational DOF
        return s_trans

    # Rotational entropy component
    s_rot = _s_rot_rr(species, temp=temp, sigma_r=params.sigma_r)

    # Vibrational entropy component
    if params.method == LFMethod.igm:
        s_vib = _igm_s_vib(species, temp)

    elif params.method == LFMethod.truhlar:
        s_vib = _truhlar_s_vib(species, temp, shift_freq=params.shift)

    elif (
        params.method == LFMethod.grimme or params.method == LFMethod.minenkov
    ):
        s_vib = _grimme_s_vib(
            species, temp, omega_0=params.w0, alpha=params.alpha
        )

    else:
        raise NotImplementedError(f"Unrecognised method: {params.method}")

    logger.info(
        f"S_trans = {s_trans*Constants.n_a:.3f} J K-1 mol-1\n"
        f"S_rot = {s_rot*Constants.n_a:.3f} J K-1 mol-1\n"
        f"S_vib = {s_vib*Constants.n_a:.3f} J K-1 mol-1\n"
        f"S_elec = 0.0"
    )

    return s_trans + s_rot + s_vib


def _zpe(species: "Species"):
    """
    Calculate the zero point energy of a molecule, contributed to by the real
    (positive) frequencies

    ---------------------------------------------------------------------------
    Arguments:
        species (autode.species.Species):

    Returns:
        (float): E_ZPE in SI units
    """

    if species.n_atoms < 2:
        return 0.0

    assert species.vib_frequencies is not None, "Must have frequenecies"

    zpe = 0.0
    for freq in species.vib_frequencies:
        zpe += 0.5 * SIConstants.h * float(freq.real.to("hz"))

    return float(zpe)


def _internal_vib_energy(species: "Species", params: _ThermoParams) -> float:
    """
    Calculate the internal energy from vibrational motion within the IGM

    ---------------------------------------------------------------------------
    Arguments:
        species (autode.species.Species):
        params (_ThermoParams):

    Returns:
        (float): U_vib in SI units
    """
    assert species.vib_frequencies is not None, "Must have frequenecies"

    temp = params.T
    w0_cm = float(params.w0.to("cm-1"))

    u = 0.0  # Total internal vibrational energy
    u_r = 0.5 * SIConstants.k_b * temp  # Free rotor internal energy

    # Final 6 vibrational frequencies are translational/rotational
    for freq in species.vib_frequencies:
        freq_cm = float(freq.real.to("cm-1"))
        x = freq_cm * Constants.c_in_cm * SIConstants.h / SIConstants.k_b

        u_v = SIConstants.k_b * x * (1.0 / (np.exp(x / temp) - 1.0))
        if params.method == LFMethod.minenkov:
            w = _grimme_w(omega_0=w0_cm, freq=freq_cm, alpha=params.alpha)
            u += w * u_v + (1 - w) * u_r
        else:
            u += u_v

    return float(u)


def _internal_energy(species: "Species", params: _ThermoParams) -> float:
    """
    Calculate the internal energy of a molecule

    ---------------------------------------------------------------------------
    Arguments:
        species (autode.species.Species):
        params (_ThermoParams):

    Returns:
        (float): U_cont in SI units
    """
    temp = params.T
    e_trns = 1.5 * SIConstants.k_b * temp

    if species.n_atoms < 2:
        # A molecule one or no atoms has no rotational/vibrational DOF
        return e_trns

    if species.is_linear():
        # Linear molecules only have two rotational degrees of freedom -> RT
        e_rot = SIConstants.k_b * temp

    else:
        # From equipartition with 3 DOF -> 3/2 RT contribution to the energy
        e_rot = 1.5 * SIConstants.k_b * temp

    zpe = _zpe(species)
    e_vib = _internal_vib_energy(species, params)

    logger.info(
        f"ZPE =     {zpe*Constants.n_a/1E3:.3f}    kJ mol-1\n"
        f"E_trans = {e_trns*Constants.n_a/1E3:.3f} kJ mol-1\n"
        f"E_rot =   {e_rot*Constants.n_a/1E3:.3f}  kJ mol-1\n"
        f"E_vib =   {e_vib*Constants.n_a/1E3:.3f}  kJ mol-1"
    )

    return zpe + e_trns + e_rot + e_vib
