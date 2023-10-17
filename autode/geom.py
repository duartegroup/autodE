import numpy as np

from typing import Sequence, Union, TYPE_CHECKING, List
from scipy.spatial.distance import cdist
from scipy.spatial import distance_matrix
from autode.log import logger

if TYPE_CHECKING:
    from autode.values import Angle
    from autode.species.species import Species
    from autode.atoms import Atoms


def are_coords_reasonable(coords: np.ndarray) -> bool:
    """
    Determine if a set of coords are reasonable. No distances can be < 0.7 Å
    and if there are more than 4 atoms ensure they do not all lie in the same
    plane. The latter possibility arises from RDKit's conformer generation
    algorithm breaking

    ---------------------------------------------------------------------------
    Arguments:
        coords (np.ndarray): Species coordinates as a n_atoms x 3 array

    Returns:
        bool:
    """
    n_atoms = len(coords)

    # Generate a n_atoms x n_atoms matrix with ones on the diagonal
    dist_mat = distance_matrix(coords, coords) + np.identity(n_atoms)

    if np.min(dist_mat) < 0.7:
        logger.warning(
            "There is a distance < 0.7 Å. Structure is *not* " "sensible"
        )
        return False

    return True


def proj(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """
    Calculate the projection of v onto the direction of u. Useful for
    https://en.wikipedia.org/wiki/Gram%E2%80%93Schmidt_process

    ---------------------------------------------------------------------------
    Arguments:
        u (np.ndarray):

        v (np.ndarray):

    Returns:
        (np.ndarray): proj_u(v)
    """
    return (np.dot(u, v) / np.dot(u, u)) * u


def get_rot_mat_kabsch(
    p_matrix: np.ndarray, q_matrix: np.ndarray
) -> np.ndarray:
    """
    Get the optimal rotation matrix with the Kabsch algorithm. Notation is from
    https://en.wikipedia.org/wiki/Kabsch_algorithm

    ---------------------------------------------------------------------------
    Arguments:
        p_matrix (np.ndarray):

        q_matrix (np.ndarray):

    Returns:
        (np.ndarray): rotation matrix
    """

    h = np.matmul(p_matrix.transpose(), q_matrix)
    u, _, vh = np.linalg.svd(h)
    d = np.linalg.det(np.matmul(vh.transpose(), u.transpose()))
    int_mat = np.identity(3)
    int_mat[2, 2] = d
    rot_matrix = np.matmul(np.matmul(vh.transpose(), int_mat), u.transpose())

    return rot_matrix


def get_rot_mat_euler_from_terms(
    a: float, b: float, c: float, d: float
) -> np.ndarray:
    """3D rotation matrix from terms unique terms in the matrix"""

    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    rot_matrix = np.array(
        [
            [aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
            [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
            [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc],
        ]
    )

    return rot_matrix


def get_rot_mat_euler(
    axis: np.ndarray, theta: Union[float, "Angle"]
) -> np.ndarray:
    """
    Compute the 3D rotation matrix using the Euler Rodrigues formula
    https://en.wikipedia.org/wiki/Euler–Rodrigues_formula
    for an anticlockwise rotation of theta radians about a given axis

    ---------------------------------------------------------------------------
    Arguments:
        axis (np.ndarray): Axis to rotate in. shape = (3,)
        theta (float): Angle in radians (float)

    Returns:
        (np.ndarray): Rotation matrix. shape = (3, 3)
    """
    if hasattr(theta, "to"):
        theta = theta.to("rad")

    axis = np.asarray(axis)
    axis = axis / np.linalg.norm(axis)  # Normalise

    a = np.cos(theta / 2.0)
    b, c, d = -axis * np.sin(theta / 2.0)
    rot_matrix = get_rot_mat_euler_from_terms(a=a, b=b, c=c, d=d)

    return rot_matrix


def get_neighbour_list(
    species: "Species",
    atom_i: int,
    index_set: Sequence[int],
) -> Sequence[int]:
    """Calculate a neighbour list from atom i as a list of atom labels

    ---------------------------------------------------------------------------
    Arguments:
        atom_i (int): index of the atom

        species (autode.species.Species):

        index_set (set(int) | None): Indexes that are possible neighbours for
                                     atom_i, if None then all atoms are ok

    Returns:
        (list(int)): list of atom ids in ascending distance away from i
    """
    if atom_i not in set(range(species.n_atoms)):
        raise ValueError(
            f"Cannot get a neighbour list for atom {atom_i} "
            f"as it is not in {species.name}, containing "
            f"{species.n_atoms} atoms"
        )

    coords = species.coordinates
    distance_vector = cdist(np.array([coords[atom_i]]), coords)[0]

    dists_and_atom_labels = {}
    for atom_j, dist in enumerate(distance_vector):
        if index_set is not None and atom_j not in index_set:
            continue

        dists_and_atom_labels[dist] = species.atoms[atom_j].label

    atom_label_neighbour_list = []
    for dist, atom_label in sorted(dists_and_atom_labels.items()):
        atom_label_neighbour_list.append(atom_label)

    return atom_label_neighbour_list


def calc_heavy_atom_rmsd(atoms1: "Atoms", atoms2: "Atoms") -> float:
    """
    Calculate the RMSD between two sets of atoms considering only the 'heavy'
    atoms, i.e. the non-hydrogen atoms

    ---------------------------------------------------------------------------
    Arguments:
        atoms1 (list(autode.atoms.Atom)):

        atoms2 (list(autode.atoms.Atom)):

    Returns:
        (float): RMSD between the two sets
    """
    if len(atoms1) != len(atoms2):
        raise ValueError(
            "RMSD must be computed between atom lists of the"
            f"same length: {len(atoms1)} =/= {len(atoms2)}"
        )

    coords1 = np.array([atom.coord for atom in atoms1 if atom.label != "H"])
    coords2 = np.array([atom.coord for atom in atoms2 if atom.label != "H"])

    if len(coords1) == 0 or len(coords2) == 0:
        logger.warning("No heavy atoms! assuming a zero RMSD")
        return 0.0

    return calc_rmsd(coords1, coords2)


def calc_rmsd(coords1: np.ndarray, coords2: np.ndarray) -> float:
    """
    Calculate the RMSD between two sets of coordinates using the Kabsch
    algorithm

    ---------------------------------------------------------------------------
    Arguments:
        coords1 (np.ndarray): shape = (n, 3)

        coords2 (np.ndarray): shape = (n ,3)

    Returns:
        (float): Root mean squared distance
    """
    assert coords1.shape == coords2.shape

    p_mat = np.array(coords2, copy=True)
    p_mat -= np.average(p_mat, axis=0)

    q_mat = np.array(coords1, copy=True)
    q_mat -= np.average(q_mat, axis=0)

    rot_mat = get_rot_mat_kabsch(p_mat, q_mat)

    fitted_coords = np.dot(rot_mat, p_mat.T).T
    return np.sqrt(np.average(np.square(fitted_coords - q_mat)))


def get_points_on_sphere(n_points: int, r: float = 1) -> List[np.ndarray]:
    """
    Find n evenly spaced points on a sphere using the "How to generate
    equidistributed points on the surface of a sphere" by Markus Deserno, 2004.

    ---------------------------------------------------------------------------
    Arguments:
        n_points (int): number of points to generate

        r (float): radius of the sphere

    Returns:
        (list(np.ndarray))
    """
    points = []

    a = 4.0 * np.pi * r**2 / n_points
    d = np.sqrt(a)
    m_theta = int(np.round(np.pi / d))
    d_theta = np.pi / m_theta
    d_phi = a / d_theta

    for m in range(m_theta):
        theta = np.pi * (m + 0.5) / m_theta
        m_phi = int(np.round(2.0 * np.pi * np.sin(theta) / d_phi))

        for n in range(m_phi):
            phi = 2.0 * np.pi * n / m_phi
            point = np.array(
                [
                    r * np.sin(theta) * np.cos(phi),
                    r * np.sin(theta) * np.sin(phi),
                    r * np.cos(theta),
                ]
            )

            points.append(point)

    return points


def symm_matrix_from_ltril(
    array: Union[Sequence[float], Sequence[Sequence[float]]]
) -> np.ndarray:
    """
    Construct a symmetric matrix from the lower triangular elements e.g.::

        array = [0, 1, 2] ->  array([[0, 1],
                                     [1, 2]])

    ---------------------------------------------------------------------------
    Arguments:
        array (list(float) | np.array):

    Returns:
        (np.ndarray):
    """

    if len(array) > 0 and type(array[0]) in (list, np.ndarray):
        # Flatten the array of arrays
        array = [item for sublist in array for item in sublist]  # type: ignore

    n = int((np.sqrt(8 * len(array) + 1) - 1) / 2)

    matrix = np.zeros(shape=(n, n), dtype="f8")

    try:
        matrix[np.tril_indices(n=n, k=0)] = np.array(array)

    except ValueError:
        raise ValueError(
            "Array was not the correct shape to be broadcast "
            "into the lower triangle. Need N(N+1)/2 elements"
            "for an NxN array"
        )

    # Symmetrise by making the upper triangular elements to the lower
    lower_idxs = np.tril_indices(n=n, k=-1)
    matrix.T[lower_idxs] = matrix[lower_idxs]

    return matrix
