import numpy as np


class Constants:
    """
    The physical constants were taken from the 2022 CODATA recommended values, https://physics.nist.gov/cuu/Constants/index.html

    """

    n_a = 6.02214076e23  # molecules mol-1

    a0 = 5.29177210544e-11  # m, uncertainty 0.00000000082e-11

    c = 299792458  # m s^-1

    m_e = 9.1093837139e-31  # kg, uncertainty 0.0000000028e-31

    ha_to_J = 4.3597447222060e-18  # Hartree^-1 J, uncertainty 0.000000000000048e-18
    J_to_ha = 1.0 / ha_to_J  # J Hartree^-1

    kcal_to_kJ = kcal2kJ = 4.184  # kJ kcal^-1

    ha_to_kJmol = ha2kJmol = ha_to_J * n_a / 1e3  # Hartree^-1 kJ mol^-1
    ha_to_kcalmol = ha2kcalmol = ha_to_kJmol / kcal_to_kJ  # Hartree^-1 kcal mol^-1

    ha_to_eV = ha2eV = 27.211386245981  # eV Hartree^-1, uncertainty 0.00000000000030
    eV_to_ha = eV2ha = 1.0 / ha_to_eV  # Hartree eV^-1

    rad_to_deg = 180 / np.pi  # deg rad^-1

    a0_to_ang = a02ang = a0 / 1e-10  # Å bohr^-1
    ang_to_a0 = ang2a0 = 1.0 / a0_to_ang  # bohr Å^-1
    ang_to_nm = 0.1  # nm ang^-1
    ang_to_pm = 100  # pm ang^-1
    ang_to_m = 1e-10  # m ang^-1
    a0_to_m = a0_to_ang * ang_to_m  # Å m^-1

    per_cm_to_hz = c_in_cm = c * 100  # cm s^-1

    amu_to_kg = 1.66053906892e-27  # kg amu^-1, uncertainty 0.00000000052e-27
    amu_to_me = amu_to_kg / m_e  # m_e amu^-1

    atm_to_pa = 101325  # Pa atm^-1
    dm_to_m = 0.1  # m dm^-1
