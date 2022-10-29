import autode as ade
from autode.methods import XTB
from autode.calculations import Calculation
from scipy.optimize import minimize
import numpy as np
from scipy.spatial import distance_matrix

ade.Config.max_num_complex_conformers = 10

ha_to_kcalmol = 627.509
ang_to_a0 = 1.0 / 0.529177


def rot_matrix(axis, theta):
    """Compute the 3D rotation matrix using:
    https://en.wikipedia.org/wiki/Euler–Rodrigues_formula"""
    axis = np.asarray(axis)
    axis /= np.linalg.norm(axis)
    a = np.cos(theta / 2.0)
    b, c, d = -axis * np.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array(
        [
            [aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
            [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
            [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc],
        ]
    )


def rotation_translation(
    x, atoms, fixed_idxs_, shift_idxs_, return_energy=True
):
    """Apply a rotation and translation"""

    f_coords = np.array([atoms[i].coord for i in fixed_idxs_], copy=True)
    f_charges = np.array([atoms[i].charge for i in fixed_idxs_])
    f_vdw = np.array([atoms[i].vdw for i in fixed_idxs_])

    s_coords = np.array([atoms[i].coord for i in shift_idxs_], copy=True)
    # Shift to ~COM
    com = np.average(s_coords, axis=0)
    s_coords -= com
    # Apply the roation
    s_coords = rot_matrix(axis=x[:3], theta=x[3]).dot(s_coords.T).T
    # Shift back, and apply the translation
    s_coords += com + x[4:]

    s_charges = np.array([atoms[i].charge for i in shift_idxs_])
    s_vdw = np.array([atoms[i].vdw for i in shift_idxs_])

    dist_mat = distance_matrix(f_coords, s_coords)

    # Matrix with the pairwise additions of the vdW radii
    sum_vdw_radii = np.add.outer(f_vdw, s_vdw)

    # Magic numbers derived from fitting potentials to noble gas dimers and
    #  plotting against the sum of vdw radii
    b_mat = 0.083214 * sum_vdw_radii - 0.003768
    a_mat = 11.576415 * (0.175541 * sum_vdw_radii + 0.316642)
    exponent_mat = -(dist_mat / b_mat) + a_mat

    repulsion = np.sum(np.exp(exponent_mat))

    # Charges are already in units of e
    prod_charge_mat = np.outer(f_charges, s_charges)

    # Compute the pairwise iteration energies as V = q1 q2 / r in atomic units
    energy_mat = prod_charge_mat / (ang_to_a0 * dist_mat)
    electrostatic = ha_to_kcalmol * np.sum(energy_mat) / 2.0

    if return_energy:
        return repulsion + electrostatic

    # Set the new coordinated of the shifted atoms, if required
    n = 0
    for i, atom in enumerate(atoms):
        if i in shift_idxs_:
            atom.coord = s_coords[n]
            n += 1

    return atoms


def set_charges_vdw(species):
    """Calculate the partial atomic charges to atoms with XTB"""
    calc = Calculation(
        name="tmp",
        molecule=species,
        method=XTB(),
        keywords=ade.SinglePointKeywords([]),
    )
    calc.run()
    charges = calc.get_atomic_charges()

    for i, atom in enumerate(species.atoms):
        atom.charge = charges[i]
        atom.vdw = float(atom.vdw_radius)

    return None


if __name__ == "__main__":

    h2o = ade.Molecule(smiles="O")
    set_charges_vdw(h2o)

    water_dimer = ade.species.NCIComplex(h2o, h2o)
    water_dimer._generate_conformers()

    shift_idxs = water_dimer.atom_indexes(mol_index=1)
    fixed_idxs = [i for i in range(water_dimer.n_atoms) if i not in shift_idxs]

    for conformer in water_dimer.conformers:

        opt = minimize(
            rotation_translation,
            x0=np.random.random(size=7),
            args=(conformer.atoms, fixed_idxs, shift_idxs),
            method="L-BFGS-B",
            tol=0.01,
        )
        print(opt)

        conformer.energy = opt.fun
        conformer.atoms = rotation_translation(
            opt.x, conformer.atoms, fixed_idxs, shift_idxs, return_energy=False
        )
        conformer.print_xyz_file()
