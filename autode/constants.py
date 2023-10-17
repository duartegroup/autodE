class Constants:
    n_a = 6.022140857e23  # molecules mol-1

    ha_to_kcalmol = ha2kcalmol = 627.509  # Hartree^-1 kcal mol^-1
    ha_to_kJmol = ha2kJmol = 2625.50  # Hartree^-1 kJ mol^-1

    ha_to_J = ha_to_kJmol * 1000 / n_a  # Hartree^-1 J
    J_to_ha = 1.0 / ha_to_J  # J Hartree^-1

    eV_to_ha = eV2ha = 0.0367493  # Hartree eV^-1
    ha_to_eV = ha2eV = 1.0 / eV_to_ha  # eV Hartree^-1

    kcal_to_kJ = kcal2kJ = 4.184  # kJ kcal^-1

    rad_to_deg = 57.29577951308232087679815  # deg rad^-1

    a0_to_ang = a02ang = 0.529177  # Å bohr^-1
    ang_to_a0 = ang2a0 = 1.0 / a0_to_ang  # bohr Å^-1
    ang_to_nm = 0.1  # nm ang^-1
    ang_to_pm = 100  # pm ang^-1
    ang_to_m = 1e-10  # m ang^-1
    a0_to_m = a0_to_ang * ang_to_m  # Å m^-1

    per_cm_to_hz = c_in_cm = 299792458 * 100  # cm s^-1

    amu_to_kg = 1.66053906660e-27  # kg amu^-1
    amu_to_me = 1822.888486209  # m_e amu^-1

    atm_to_pa = 101325  # Pa atm^-1
    dm_to_m = 0.1  # m dm^-1
