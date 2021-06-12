"""
Thermochemistry calculation from frequencies and coordinates. Copied from
otherm (https://github.com/duartegroup/otherm) 16/05/2021
"""
import numpy as np
from autode.config import Config
from autode.constants import Constants


class SIConstants:
    k_b = 1.38064852E-23                # J K-1
    h = 6.62607004E-34                  # J s
    n_a = 6.022140857E23                # molecules mol-1
    r = k_b * n_a                       # J K-1 mol-1
    c = 299792458                       # m s-1


def calculate_thermo_cont(species, temp=298.15):
    """
    Set the thermochemical contributions (Enthalpic, Free Energy) of a
    species

    Arguments:
        species (autode.species.Species):

    Keyword Arguments:
        temp (float): Temperature in K
    """

    s_cont = calc_entropy(species,
                          method=Config.lfm_method,
                          temp=temp,
                          ss=Config.standard_state,
                          shift=Config.vib_freq_shift,
                          w0=Config.grimme_w0,
                          alpha=Config.grimme_alpha)

    u_cont = calc_internal_energy(species, temp=temp)
    h_cont = u_cont + SIConstants.r * temp
    g_cont = h_cont - temp * s_cont

    return None


def calc_q_trans_igm(molecule, ss, temp):
    """
    Calculate the translational partition function using the PIB model,
    coupled with an effective volume

    :param molecule: (otherm.Molecule)
    :param ss: (str) Standard state to use {1M, 1atm}
    :param temp: (float) Temperature in K
    :return: (float) Translational partition function q_trns
    """

    if ss.lower() == '1atm':
        effective_volume = Constants.k_b * temp / Constants.atm_to_pa

    elif ss.lower() == '1m':
        effective_volume = 1.0 / (Constants.n_a * (1.0 / Constants.dm_to_m)**3)

    else:
        raise NotImplementedError

    q_trans = ((2.0 * np.pi * molecule.mass * Constants.k_b * temp /
                Constants.h**2)**1.5 * effective_volume)

    return q_trans


def calc_q_rot_igm(molecule, temp):
    """
    Calculate the rotational partition function using the IGM method. Uses the
    rotational symmetry number, default = 1

    :param molecule: (otherm.Molecule)
    :param temp: (float) Temperature in K
    :return: (float) Rotational partition function q_rot
    """

    i_mat = calc_moments_of_inertia(molecule.xyzs)
    omega = Constants.h**2 / (8.0 * np.pi**2 * Constants.k_b * i_mat)

    if molecule.n_atoms == 1:
        return 1

    else:
        # Product of the diagonal elements
        omega_prod = omega[0, 0] * omega[1, 1] * omega[2, 2]
        return temp**1.5/molecule.sigma_r * np.sqrt(np.pi / omega_prod)


def calc_q_vib_igm(molecule, temp):
    """
    Calculate the vibrational partition function using the IGM method.
    Uses the rotational symmetry number, default = 1

    :param molecule: (otherm.Molecule)
    :param temp: (float) Temperature in K
    :return: (float) Vibrational partition function q_rot
    """

    if molecule.n_atoms == 1:
        molecule.q_vib = 1
        return molecule.q_vib

    for freq in molecule.real_vib_freqs():
        x = freq * Constants.c_in_cm * Constants.h / Constants.k_b
        molecule.q_vib *= np.exp(-x / (2.0 * temp)) / (1.0 - np.exp(-x / temp))

    return molecule.q_vib


def calc_s_trans_pib(molecule, ss, temp):
    """Calculate the translational entropy using a particle in a box model

    :param molecule: (otherm.Molecule)
    :param ss: (str) Standard state to use for calculating the effective box
               size in the q_trans calculation
    :param temp: (float) Temperature in K
    :return: (float) S_trans in J K-1 mol-1
    """

    q_trans = calc_q_trans_igm(molecule, ss=ss, temp=temp)
    return Constants.r * (np.log(q_trans) + 1.0 + 1.5)


def calc_s_rot_rr(molecule, temp):
    """
    Calculate the rigid rotor (RR) entropy

    :param molecule: (otherm.Molecule)
    :return: (float) S_rot in J K-1 mol-1
    """

    if molecule.n_atoms == 1:
        return 0

    q_rot = calc_q_rot_igm(molecule, temp=temp)

    if molecule.is_linear():
        return Constants.r * (np.log(q_rot) + 1.0)

    else:
        return Constants.r * (np.log(q_rot) + 1.5)


def calc_igm_s_vib(molecule, temp):
    """
    Calculate the entropy of a molecule according to the Ideal Gas Model (IGM)
    RRHO method

    :param molecule: (otherm.Molecule)
    :param temp: (float) Temperature in K
    :return: (float) S_vib in J K-1 mol-1
    """
    s = 0.0

    for freq in molecule.real_vib_freqs():
        x = freq * Constants.c_in_cm * Constants.h / (Constants.k_b * temp)
        s += Constants.r * ((x / (np.exp(x) - 1.0)) - np.log(1.0 - np.exp(-x)))

    return s


def calc_truhlar_s_vib(molecule, temp, shift_freq):
    """
    Calculate the entropy of a molecule according to the Truhlar's method of
    shifting low frequency modes

    :param molecule: (otherm.Molecule)
    :param temp: (float) Temperature in K
    :param shift_freq: (float) Shift all frequencies to this value
    :return: (float) S_vib in J K-1 mol-1
    """
    s = 0

    for freq in molecule.real_vib_freqs():

        # Threshold lower bound of the frequency
        freq = max(freq, shift_freq)

        x = freq * Constants.c_in_cm * Constants.h / Constants.k_b
        s += Constants.r * (((x / temp) / (np.exp(x / temp) - 1.0)) -
                                np.log(1.0 - np.exp(-x / temp)))

    return s


def calc_grimme_s_vib(species, temp, omega_0, alpha):
    """
    Calculate the entropy according to Grimme's qRRHO method of RR-HO
    interpolation in Chem. Eur. J. 2012, 18, 9955

    :param species: (otherm.Molecule)
    :param temp: (float) Temperature in K
    :param omega_0: (float) ω0 parameter
    :param alpha: (float) α parameter
    :return: (float) S_vib in J K-1 mol-1
    """
    s = 0.0

    i_mat = species.moment_of_inertia

    # Average I = (I_xx + I_yy + I_zz) / 3.0
    b_avg = np.trace(i_mat) / 3.0

    for freq in species.vib_frequencies:

        omega = freq.real * Constants.c_in_cm

        mu = SIConstants.h / (8.0 * np.pi**2 * omega)
        mu_prime = (mu * b_avg) / (mu + b_avg)

        x = omega * SIConstants.h / (SIConstants.k_b * temp)
        s_v = SIConstants.r * ((x / (np.exp(x) - 1.0)) - np.log(1.0 - np.exp(-x)))
        s_r = SIConstants.r * (0.5 + np.log(np.sqrt((8.0 * np.pi**3 * mu_prime * SIConstants.k_b * temp) /
                                                  (SIConstants.h**2)
                                                  )))

        w = 1.0 / (1.0 + (omega_0 / freq)**alpha)

        s += w * s_v + (1.0 - w) * s_r

    return s


def calc_entropy(species, method, temp, ss, shift, w0, alpha):
    """
    Calculate the entropy


    """

    # Translational entropy component
    s_trans = calc_s_trans_pib(species, ss=ss, temp=temp)

    if species.n_atoms == 1:
        # A molecule with only one atom has no rotational/vibrational DOF
        return s_trans

    # Rotational entropy component
    s_rot = calc_s_rot_rr(species, temp=temp)

    # Vibrational entropy component
    if method.lower() == 'igm':
        s_vib = calc_igm_s_vib(species, temp)

    elif method.lower() == 'truhlar':
        s_vib = calc_truhlar_s_vib(species, temp, shift_freq=shift)

    elif method.lower() == 'grimme':
        s_vib = calc_grimme_s_vib(species, temp, omega_0=w0, alpha=alpha)

    else:
        raise NotImplementedError(f'Unrecognised method: {method}')

    return s_trans + s_rot + s_vib


def calc_zpe(species):
    """
    Calculate the zero point energy of a molecule, contributed to by the real
    (positive) frequencies

    Arguments:
        species (autode.species.Species):

    Returns:
        (float): E_ZPE
    """

    zpe = 0.0

    for freq in species.vib_frequencies:
        zpe += 0.5 * SIConstants.h * Constants.n_a * freq.real.to('s-1')

    return zpe


def calc_internal_vib_energy(species, temp):
    """
    Calculate the internal energy from vibrational motion within the IGM

    Arguments:
        species (autode.species.Species):
        temp (float): Temperature in K

    Returns:
        (float): U_vib in SI units
    """
    e_vib = 0.0

    # Final 6 vibrational frequencies are translational/rotational
    for freq in species.vib_frequencies:
        x = freq.real * Constants.c_in_cm * SIConstants.h / SIConstants.k_b
        e_vib += SIConstants.r * x * (1.0 / (np.exp(x/temp) - 1.0))

    return e_vib


def calc_internal_energy(species, temp):
    """
    Calculate the internal energy of a molecule


    Arguments:
        species (autode.species.Species):
        temp (float): Temperature in K

    Returns:
        (float): U_cont in SI units
    """
    zpe = calc_zpe(species)
    e_trns = 1.5 * SIConstants.r * temp

    if species.is_linear():
        # Linear molecules only have two rotational degrees of freedom -> RT
        e_rot = SIConstants.r * temp

    else:
        # From equipartition with 3 DOF -> 3/2 RT contribution to the energy
        e_rot = 1.5 * SIConstants.r * temp

    e_vib = calc_internal_vib_energy(species, temp=temp)

    return zpe + e_trns + e_rot + e_vib
