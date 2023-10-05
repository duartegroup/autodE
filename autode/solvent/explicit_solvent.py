import numpy as np
from typing import Optional, TYPE_CHECKING, Any, List
from scipy.spatial import distance_matrix

from autode.geom import get_points_on_sphere, get_rot_mat_euler
from autode.log import logger
from autode.atoms import AtomCollection
from autode.solvent.solvents import Solvent

if TYPE_CHECKING:
    from autode.species.species import Species


class _RandomPointGenerator:
    r"""
    Generator for points (unit vectors) in solvent shells. e.g. where if x is
    a solute molecule the vectors in the different shells::

            ------
                   `
            ---      `
                \     |
            x    |    |

    """

    def __init__(self, random_state: np.random.RandomState):
        """
        Point generator

        Arguments:
            random_state (numpy.random.mtrand.RandomState):
        """
        self.random_state = random_state
        self._sphere_n = 1
        self._points: List[np.ndarray] = []

    def random_point(self) -> np.ndarray:
        """
        Generate a random point in a solvent shell. Will return points on
        the surface of the solvent shell (self._sphere_n) and increment the
        solvent shell when there are none left

        Returns:
            (np.ndarray): Point on the 3D sphere
        """

        if len(self._points) == 0:
            # Surface area of the sphere scales r^2, so square solvent shell
            self._points = get_points_on_sphere(
                n_points=self._sphere_n**2 * 10
            )
            self._sphere_n += 1

        idx = self.random_state.randint(0, len(self._points))
        return self._points.pop(idx)


class ExplicitSolvent(AtomCollection, Solvent):
    """Explicit solvation"""

    def __init__(
        self,
        solvent: "Species",
        num: int,
        solute: Optional["Species"] = None,
        **kwargs,
    ):
        """
        Explicit solvent. Initial construction attempts to generate a
        reasonable distribution around the (unmodified) solute. Only supports
        unicomponent uncharged solvents.

        ----------------------------------------------------------------------
        Arguments:

            solvent (autode.species.species.Species): Solvent molecule (copied)

            num (int): Number of solvent molecules to add


        Keyword Arguments:

            solute (autode.species.species.Species | None): Solute which this
                   solvent surrounds. If None then no translation to the
                   explicit solvent molecules will be applied

            aliases (list(str)): List of aliases of this solvent
        """
        if num <= 0:
            raise ValueError(
                "Must solvate with at least a single solvent "
                f"molecule. Had {num}"
            )

        solvent_atoms = sum((solvent.atoms.copy() for _ in range(num)), None)  # type: ignore
        AtomCollection.__init__(self, atoms=solvent_atoms)
        Solvent.__init__(
            self,
            name=solvent.name,
            smiles=None,
            aliases=kwargs.get("aliases", None),
        )

        self.solvent_n_atoms = solvent.n_atoms
        # TODO: Something better than this hardcoded value
        self.solvent_radius = solvent.radius.to("ang") + 2.0

        if solute is not None:
            self.randomise_around(solute)

    def __eq__(self, other: Any) -> bool:
        """Equality between two explicit solvent environments"""

        if (
            isinstance(other, ExplicitSolvent)
            and self.n_atoms == other.n_atoms
        ):
            assert self.atoms and other.atoms  # keep mypy happy
            return all(
                o_at.label == at.label
                for o_at, at in zip(other.atoms, self.atoms)
            )

        return False

    @property
    def is_implicit(self) -> bool:
        """Is this solvent implicit?

        Returns:
            (bool): False
        """
        return False

    @property
    def n_solvent_molecules(self) -> int:
        """Number of solvent molecules comprising this explicit solvent
        cluster

        Returns:
            (int): n
        """
        return self.n_atoms // self.solvent_n_atoms

    def solvent_atom_idxs(self, i: int) -> np.ndarray:
        """
        Atom indexes of an particular solvent molecule

        Returns:
            (np.ndarray): Atom indexes
        """
        if i < 0 or i >= self.n_solvent_molecules:
            raise ValueError(
                f"Cannot find the indexes for the {i}th solvent "
                f"only had {self.n_solvent_molecules}."
            )

        first_idx = i * self.solvent_n_atoms
        last_idx = first_idx + self.solvent_n_atoms

        return np.array(range(first_idx, last_idx), dtype=int)

    @staticmethod
    def _too_close_to_solute(
        solvent_coords: np.ndarray,
        solute_coords: np.ndarray,
        solute_radius: float,
    ) -> bool:
        """
        Are a set of solvent coordinates too close to the solute? (for a
        particular solute radius)

        Arguments:
            solvent_coords (np.ndarray):  Shape = (N, 3)

            solute_coords (np.ndarray):  Shape = (M, 3)

            solute_radius (float): Radius (Å)
        """
        min_dist = np.min(distance_matrix(solute_coords, solvent_coords))
        return min_dist < solute_radius

    def _too_close_to_solvent(
        self, coords: np.ndarray, solvent_idxs: np.ndarray, max_idx: int
    ) -> bool:
        """
        Are a set of solvent coordinates too close to the solvent molecules
        that have already been translated?

        Arguments:
            coords (np.ndarray):  Shape = (N, 3) Coordinates of all
                                  the solvent molecules

            solvent_idxs (np.ndarray): Integer array of atom indexes of a
                                       particular solvent molecule

            max_idx (int): Indexes up to which the repulsion should be
                                calculated. NOT INCLUSIVE
        """
        if max_idx == 0:
            return False

        min_dist = np.min(
            distance_matrix(
                coords[solvent_idxs], coords[: max_idx * self.solvent_n_atoms]
            )
        )

        return min_dist < self.solvent_radius

    def randomise_around(self, solute: "Species") -> None:
        r"""
        Randomise the positions of the solvent molecules around the solute,
        for example using a methane solute and water solvent::


                               H2O
                        H20
                 H2o            H2O
                                        H2O
               H2O      CH4      H2O

                     H2O      H2O


        where the solvent molecules are roughly packed in shells around the
        solute.

        Arguments:
            solute (autode.species.species.Species):
        """
        logger.info(
            f"Randomising {self.n_solvent_molecules} solvent "
            f"molecules around {solute}"
        )

        coords = self.coordinates
        assert (
            coords is not None
        ), "Must have coordinates to populate solvent around"

        # ----------------- Properties of the solute molecule -----------------
        m_radius = solute.radius.to("ang") + 1.0  # Assume some exterior H
        m_origin = np.average(solute.coordinates, axis=0)
        m_coords = solute.coordinates - m_origin
        # ---------------------------------------------------------------------

        rand = np.random.RandomState()
        pg = _RandomPointGenerator(random_state=rand)

        for i in range(self.n_solvent_molecules):
            idxs = self.solvent_atom_idxs(i)
            coords[idxs] -= np.average(coords[idxs], axis=0)  # -> origin

            # Apply a random rotation to the solvent molecule
            rand_rot_mat = get_rot_mat_euler(
                axis=rand.uniform(-1.0, 1.0, size=3),
                theta=rand.uniform(-np.pi, np.pi),
            )

            coords[idxs] = np.dot(coords[idxs], rand_rot_mat.T)

            # Select a random vector along which this solvent molecule is to be
            # translated until there is not any close contacts
            vec = 0.1 * pg.random_point()

            while self._too_close_to_solute(
                coords[idxs], m_coords, m_radius
            ) or self._too_close_to_solvent(coords, idxs, i):
                coords[idxs] += vec

        # Finally, translate to be centred around the solute's origin
        self.coordinates = coords + m_origin
        return None
