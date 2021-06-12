"""
Thermochemistry calculation from frequencies and coordinates. Copied from
otherm (https://github.com/duartegroup/otherm) 16/05/2021
"""
import numpy as np
from autode.constants import Constants


class SIConstants:
    k_b = 1.38064852E-23                # J K-1
    h = 6.62607004E-34                  # J s
    n_a = 6.022140857E23                # molecules mol-1
    r = k_b * n_a                       # J K-1 mol-1
    c = 299792458                       # m s-1


def set_thermo_cont(species, temp):
    """
    Set the thermochemical contributions (Enthalpic, Free Energy) of a
    species

    Arguments:
        species (autode.species.Species):

        temp (float): Temperature in K
    """


    raise NotImplementedError


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


def calc_grimme_s_vib(molecule, temp, omega_0, alpha):
    """
    Calculate the entropy according to Grimme's qRRHO method of RR-HO
    interpolation in Chem. Eur. J. 2012, 18, 9955

    :param molecule: (otherm.Molecule)
    :param temp: (float) Temperature in K
    :param omega_0: (float) ω0 parameter
    :param alpha: (float) α parameter
    :return: (float) S_vib in J K-1 mol-1
    """
    s = 0.0

    i_mat = calc_moments_of_inertia(molecule.xyzs)

    # Average I = (I_xx + I_yy + I_zz) / 3.0
    b_avg = np.trace(i_mat) / 3.0

    for freq in molecule.real_vib_freqs():

        omega = freq * Constants.c_in_cm
        mu = Constants.h / (8.0 * np.pi**2 * omega)
        mu_prime = (mu * b_avg) / (mu + b_avg)

        x = freq * Constants.c_in_cm * Constants.h / (Constants.k_b * temp)
        s_v = Constants.r * ((x / (np.exp(x) - 1.0)) - np.log(1.0 - np.exp(-x)))
        s_r = Constants.r * (0.5 + np.log(np.sqrt((8.0 * np.pi**3 * mu_prime * Constants.k_b * temp) /
                                                  (Constants.h**2)
                                                  )))

        w = 1.0 / (1.0 + (omega_0 / freq)**alpha)

        s += w * s_v + (1.0 - w) * s_r

    return s


def calc_entropy(molecule, method='grimme', temp=298.15, ss='1M',
                 shift=100, w0=100, alpha=4):
    """
    Calculate the entropy

    :param molecule: (otherm.Molecule)

    :return: (float) S in J K-1 mol-1
    """

    # Translational entropy component
    s_trans = calc_s_trans_pib(molecule, ss=ss, temp=temp)

    if molecule.n_atoms == 1:
        # A molecule with only one atom has no rotational/vibrational DOF
        return s_trans

    # Rotational entropy component
    s_rot = calc_s_rot_rr(molecule, temp=temp)

    # Vibrational entropy component
    if method.lower() == 'igm':
        s_vib = calc_igm_s_vib(molecule, temp)

    elif method.lower() == 'truhlar':
        s_vib = calc_truhlar_s_vib(molecule, temp, shift_freq=shift)

    elif method.lower() == 'grimme':
        s_vib = calc_grimme_s_vib(molecule, temp, omega_0=w0, alpha=alpha)

    else:
        raise NotImplementedError

    return s_trans + s_rot + s_vib


def calc_zpe(molecule):
    """
    Calculate the zero point energy of a molecule, contributed to by the real
    (positive) frequencies

    :param molecule: (otherm.Molecule)
    :return: (float) E_ZPE
    """

    zpe = 0.0

    for freq in molecule.real_vib_freqs():
        zpe += 0.5 * Constants.h * Constants.n_a * Constants.c_in_cm * freq

    return zpe


def calc_internal_vib_energy(molecule, temp):
    """
    Calculate the internal energy from vibrational motion within the IGM

    :param molecule: (otherm.Molecule)
    :param temp: (float)
    :return: (float) U_vib
    """

    e_vib = 0.0

    # Final 6 vibrational frequencies are translational/rotational
    for freq in molecule.real_vib_freqs():
        x = freq * Constants.c_in_cm * Constants.h / Constants.k_b
        e_vib += Constants.r * x * (1.0 / (np.exp(x/temp) - 1.0))

    return e_vib


def calc_internal_energy(molecule, temp):
    """
    Calculate the internal energy of a molecule

    :param molecule: (otherm.Molecule)
    :param temp: (float) Temperature in K
    :return: (float) U
    """

    zpe = calc_zpe(molecule)
    e_trns = 1.5 * Constants.r * temp

    if molecule.is_linear():
        # Linear molecules only have two rotational degrees of freedom -> RT
        e_rot = Constants.r * temp

    else:
        # From equipartition with 3 DOF -> 3/2 RT contribution to the energy
        e_rot = 1.5 * Constants.r * temp

    e_vib = calc_internal_vib_energy(molecule, temp=temp)

    return molecule.e + zpe + e_trns + e_rot + e_vib


class Molecule:

    def shift_to_com(self):
        """Shift a molecules xyzs to the center of mass"""

        shifted_xyzs = []

        for xyz_line in self.xyzs:
            pos = np.array(xyz_line[1:]) - self.com
            shifted_xyzs.append(xyz_line[:1] + pos.tolist())

        self.xyzs = shifted_xyzs
        self.com = self.calculate_com()
        return None

    def real_vib_freqs(self):
        """Return the real (positive) vibrational frequencies"""
        # Vibrational frequencies are all but the 6 smallest (rotational +
        # translational) and also remove the largest imaginary frequency if
        # this species is a transtion state
        excluded_n = 7 if self.is_ts else 6

        # Frequencies are sorted high -> low(negative)
        if self.real_freqs:
            return [np.abs(freq) for freq in self.freqs[:-excluded_n]]

        else:
            return [freq for freq in self.freqs[:-excluded_n] if freq > 0]

    def calculate_mass(self):
        """Calculate the molecular mass of this molecule in kg"""

        atomic_symbols = [xyz[0] for xyz in self.xyzs]
        masses_amu = [Constants.atomic_masses[elm] for elm in atomic_symbols]

        return Constants.amu_to_kg * sum(masses_amu)

    def calculate_com(self):
        """
        Calculate the center of mass (COM

        :return: (np.ndarray) COM vector
        """
        total_mass = self.mass / Constants.amu_to_kg

        com_vec = np.zeros(3)  # Blank 3D vector for COM vector

        for xyz_line in self.xyzs:
            r_vec = np.array(xyz_line[1:])  # Vector for that atom
            mass = Constants.atomic_masses[xyz_line[0]]
            com_vec += (1.0 / total_mass) * mass * r_vec

        return com_vec



    def coords(self):
        """Return a numpy array shape (n_atoms, 3) of (x,y,z) coordinates"""
        return np.array([np.array(line[1:4]) for line in self.xyzs])

    def calculate_thermochemistry(self,
                                  temp=298.15,
                                  ss='1M',
                                  method='grimme',
                                  shift=100,
                                  w0=100,
                                  alpha=4,
                                  calc_sym=True,
                                  symm_n=None):
        """
        Calculate thermochemical components and the energies U, H, S, G

        -----------------------------------------------------------------------
        :param temp: (float) Temperature in K

        :param ss: (str) standard state e.g. 1M or 1atm

        :param method: (str) Method to calculate the entropy

        :param shift: (float) Shift frequency used in the Truhlar method of
                      calculating vibrational entropy. All harmonic freqencies
                      below this value will be shifted to this value

        :param w0: (float) ω0 parameter in the Grimme vibrational entropy
                   method

        :param alpha: (float) α parameter the Grimme vibrational entropy
                   method

        :param calc_sym: (bool) Force the calculation of symmetry

        :param symm_n: (int) Override the calculated symmetry number
        """
        assert len(self.freqs) == 3 * len(self.xyzs)

        # If the calculation of rotational symmetry number σR is requested or
        # there aren't too many atoms
        if calc_sym or self.n_atoms < 50:
            self.sigma_r = calc_symmetry_number(self)

        # Allow overwriting σR
        if symm_n:
            self.sigma_r = symm_n

        self.s = calc_entropy(self, method, temp, ss, shift, w0, alpha)
        self.u = calc_internal_energy(self, temp)
        self.h = self.u + Constants.r * temp
        self.g = self.h - temp * self.s

        return None

    def __init__(self, filename, is_ts=False, real_freqs=True):
        """
        Molecule initialised from an ORCA output file

        :param filename: (str)
        :param is_ts: (bool) Is this species a TS? if so then exclude
        """
        # Is this molecule a transition state, and so should have one imaginary
        # (negative frequency)
        self.is_ts = is_ts

        # Should all non-TS frequencies be made real (positive)
        self.real_freqs = real_freqs

        # Harmonic vibrational frequencies in cm-1
        self.freqs = extract_frequencies(filename)

        # Atom positions [[atom, x, y, z], ...] x/y/z in Å
        self.xyzs = extract_xyzs(filename)
        self.n_atoms = len(self.xyzs)

        # Mass in kg
        self.mass = self.calculate_mass()

        # Matrix of I values in kg m^2
        self.moments_of_inertia = calc_moments_of_inertia(self.xyzs)

        # Centre of mass np.array shape (3,) x/y/z in Å
        self.com = self.calculate_com()
        self.shift_to_com()

        # Rotational symmetry number
        self.sigma_r = 1

        # Total electronic (E), internal (U), enthalpy (H), entropy (S) and
        # Gibbs (free) energy (G) all in molar SI units
        self.e = extract_final_electronic_energy(filename)
        self.u = None
        self.h = None
        self.s = None
        self.g = None


if __name__ == '__main__':

    mol = Molecule(args.filename,
                   is_ts=args.transition_state,
                   real_freqs=args.real_freqs)
