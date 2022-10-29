import numpy as np
from autode.log import logger
from scipy.spatial import distance_matrix
from autode.geom import get_rot_mat_kabsch, get_rot_mat_euler_from_terms


class AtomType:
    def __init__(self, site_coords, is_chiral=False):
        """Base atom type class

        -----------------------------------------------------------------------
        Arguments:
            site_coords (list(np.ndarray)): Shape = (n, 3) should contain a
                        list of unit vectors pointing in directions where other
                        atoms can be added

        Keyword Arguments:
            is_chiral (bool): Is this atom type chiral e.g. a tetrahedral atom
                              with four different substituents
        """
        self.template_site_coords = np.copy(site_coords)
        self._site_coords = site_coords

        self.is_chiral = is_chiral
        self.rotate_randomly()

    @property
    def n_empty_sites(self):
        """Number of empty sites on this template"""
        return len(self._site_coords)

    def empty_site(self):
        """Iterator for the coordinate of the next free site"""
        return self._site_coords.pop(0)

    def empty_site_mr(self, point, other_coords):
        """Return the site on this atom that is furthest from all other
        coordinates using a simple 1/r potential where r is the distance from
        the site to the other coordinates

        -----------------------------------------------------------------------
        Arguments:
            point (np.ndarray): Coordinate of this atom, shape = (3,)

            other_coords (np.ndarray): Other coordinates, shape = (N, 3)

        Returns:
            (np.ndarray): Coordinate of the site centered at the origin
        """
        dists = np.array(
            [
                np.linalg.norm(other_coords - (site + point), axis=1)
                for site in self._site_coords
            ]
        )

        repulsion = np.sum(np.power(dists, -1), axis=1)
        return self._site_coords.pop(np.argmin(repulsion))

    def reset_onto(self, points, coord):
        """
        Reset the site coordinates given a set of points. Ignore any points
        located exactly at the origin and, once fitted, remove the sites
        that are coincident with the points

        -----------------------------------------------------------------------
        Arguments:
            points (iterable(np.ndarray)): List (or iterable) of points that
                   that the sites need to be reset onto

            coord (np.ndarray): Coordinate of this atom
        """
        origin = np.zeros(3)
        points = np.array(
            [
                (point - coord) / np.linalg.norm(point - coord)
                for point in points
                if not np.allclose(point, origin)
            ]
        )

        # Take a copy of the template coordinates to rotate and delete
        site_coords = np.copy(self.template_site_coords)

        if len(site_coords) == len(points) or len(points) == 0:
            logger.info("No reset needed - sites were all occupied")
            return

        logger.info(
            f"Rotating {len(site_coords)} sites onto" f" {len(points)} points"
        )

        # Rotate all the sites such that n sites are optimally orientated onto
        # the (fixed) points
        rot_mat = get_rot_mat_kabsch(
            p_matrix=site_coords[: len(points)], q_matrix=points
        )

        site_coords = np.dot(rot_mat, site_coords.T).T

        # For each point (row) calculate the minimum distance to a site on
        # this atom
        min_dists = np.min(distance_matrix(site_coords, points), axis=1)

        # Re-populate the empty sites, which are the sites that are not the
        # closest to the points
        self._site_coords = [
            coord
            for i, coord in enumerate(site_coords)
            if i not in np.argsort(min_dists)[: len(points)]
        ]
        return None

    def rotate_empty_onto(self, point, coord):
        """Rotate the site coordinates such that an empty site is coincident
        with the vector from a coordinate to a point, and remove the site
        from the list of available sites"""
        return self.rotate_onto(point, coord, site=self.empty_site())

    def rotate_randomly(self):
        """Rotate the sites randomly to prevent zero cross products"""
        point = np.copy(self._site_coords[0])
        point += np.random.uniform(0.01, 0.02, size=3)

        self.rotate_onto(
            point=point, coord=np.zeros(3), site=self._site_coords[0]
        )
        return

    def rotate_onto(self, point, coord, site):
        """
        Rotate this atom type so a site is coincident with a point if this
        atom is at a coord i.e.::

                               site
                               /
                              /         -->
            point--------coord                 point--site--coord


        -----------------------------------------------------------------------
        Arguments:
            point (np.ndarray): shape = (3,)

            coord (np.ndarray): shape = (3,)

            site (np.ndarray): shapte = (3,)
        """
        vector = point - coord

        normal = np.cross(site, vector)
        normal /= np.linalg.norm(normal)

        # Sites are normal vectors, no no need for mod
        arg = np.dot(site, vector) / np.linalg.norm(vector)

        # cos(-θ/2) = √(arg + 1) / √2
        a = np.sqrt(1.0 + arg) / np.sqrt(2)

        # sin(-θ/2) = √(1-arg) / √2
        b, c, d = -normal * (np.sqrt(1.0 - arg) / np.sqrt(2))

        # 3D rotation matrix from the Euler–Rodrigues formula
        rot_matrix = get_rot_mat_euler_from_terms(a=a, b=b, c=c, d=d)

        # Rotate all the sites (no need to translate as they're already
        # positioned around the origin)
        self._site_coords = [
            np.matmul(rot_matrix, site) for site in self._site_coords
        ]
        return None


class TerminalAtom(AtomType):
    def __init__(self):
        r"""
        Terminal atom with a site pointing along the x-axis::

                    Atom---
        """
        site_coords = [np.array([1.0, 0.0, 0.0])]

        super().__init__(site_coords)


class LinearAtom(AtomType):
    def __init__(self):
        r"""
        Linear atom with sites pointing along the x-axis::

                    ---Atom---

        WARNING: Completely linear cause failures for internal coordinate
                 optimisation algorithms and have undefined cross products.
                 Site coordinates are therefore not precisely linear
        """
        site_coords = [np.array([1.0, 0.0, 0.0]), np.array([-1.0, 0.001, 0.0])]

        super().__init__(site_coords)


class BentAtom(AtomType):
    def __init__(self):
        r"""
        Bent atom with sites generated by optimisation of H2O::

                       Atom
                     /     \
        """
        site_coords = [
            np.array([-0.78226654, -0.62294387, 0.0]),
            np.array([0.78322832, -0.62173419, 0.0]),
        ]

        super().__init__(site_coords)


class TrigonalPyramidalAtom(AtomType):
    def __init__(self):
        r"""
        Trigonal pyramidal atom e.g. P in PH3, obtained from optimisation of
        ammonia (then normalising  NH distances to 1 Å)::

                      Atom
                    /  |  \
        """
        site_coords = [
            np.array([0.90023489, -0.14794295, -0.40949973]),
            np.array([-0.58738609, -0.70512041, -0.39721881]),
            np.array([-0.32432922, 0.85865859, -0.39688283]),
        ]

        super().__init__(site_coords)


class TrigonalAtom(AtomType):
    def __init__(self):
        r"""
        Trigonal atom e.g. [CR3]+ , obtained from optimisation of BH3
        (then normalising  NH distances to 1 Å)::

                               /
                       --- Atom
                              \
        """
        site_coords = [
            np.array([-0.506363095, -0.862320319, 0.0]),
            np.array([-0.495155944, 0.868804058, 0.0]),
            np.array([0.999977780, -0.006666131, 0.0]),
        ]

        super().__init__(site_coords)


class TetrahedralAtom(AtomType):
    def __init__(self):
        r"""
        Tetrahedral atom with sites generated by optimisation of methane
        (then normalising CH distances to 1 Å)::

                         |  /
                        Atom
                      /     \

        """
        site_coords = [
            np.array([-0.404709, 0.86798519, -0.28777090]),
            np.array([-0.580775, -0.75435372, -0.30602419]),
            np.array([0.0763827, -0.01927872, 0.99689218]),
            np.array([0.9089159, -0.09390161, -0.40626889]),
        ]

        super().__init__(site_coords)


class TetrahedralNAtom(TetrahedralAtom):
    """A 'normal' order chiral tetrahedral atom"""

    def __init__(self):
        super().__init__()
        self.is_chiral = True


class TetrahedralIAtom(TetrahedralAtom):
    """An 'inverted' order chiral tetrahedral atom"""

    def empty_site(self):
        """Swap the first two yielded site coordinates, effectively swapping
        the chirality this atom's neighbours are added"""

        if len(self._site_coords) == 3:
            return self._site_coords.pop(1)

        else:
            return super().empty_site()

    def __init__(self):
        super().__init__()
        self.is_chiral = True


class SquarePlanarAtom(AtomType):
    def __init__(self):
        r"""
        Square planar atom with sites generated by optimisation of XeF4::

                       |
                   -- Atom --
                       |
        """
        site_coords = [
            np.array([-0.99779169, 0.06642094, 0.0]),
            np.array([0.06641523, 0.99779207, 0.0]),
            np.array([0.99779219, -0.06641349, 0.0]),
            np.array([-0.06642889, -0.99779116, 0.0]),
        ]

        super().__init__(site_coords)


class TrigonalBipyramidalAtom(AtomType):
    def __init__(self):
        r"""
        Trigonal bipyramidal atom with sites generated by optimisation of
        [Cn(Cl)5]-::

                        |
                        |
                  --- Atom --
                    /  |
                       |
        """
        site_coords = [
            np.array([-0.96060076, 0.0333159, -0.27592795]),
            np.array([0.27025683, -0.50748281, 0.8181824]),
            np.array([0.68796654, 0.4787655, -0.54542243]),
            np.array([-0.13392744, 0.82131359, 0.55453352]),
            np.array([0.13543747, -0.82200953, -0.55313383]),
        ]

        super().__init__(site_coords)


class OctahedralAtom(AtomType):
    def __init__(self):
        r"""
        Octahedral atom with sites generated by optimisation of [Co(Cl)6]3-::

                        |
                        | /
                  --- Atom --
                    /  |
                       |
        """
        site_coords = [
            np.array([0.06037748, 0.86107926, 0.50487332]),
            np.array([-0.96772781, -0.0717284, 0.24157384]),
            np.array([-0.06059905, -0.86084488, -0.50524632]),
            np.array([-0.2330902, 0.50187662, -0.83293986]),
            np.array([0.23246809, -0.50140995, 0.83339465]),
            np.array([0.96764951, 0.07148532, -0.24195925]),
        ]

        super().__init__(site_coords)


class PentagonalBipyramidalAtom(AtomType):
    def __init__(self):
        """Approximate trigonal pentagonal geometry by optimisation of IF7"""

        site_coords = [
            np.array([-0.82513358, 0.19948399, 0.52854584]),
            np.array([0.36100434, 0.82474278, 0.4352875]),
            np.array([-0.2989535, -0.86729114, -0.39803628]),
            np.array([0.07965322, -0.50323863, 0.86046862]),
            np.array([-0.68679889, 0.35535251, -0.63405984]),
            np.array([0.92361199, -0.38311825, 0.0127003]),
            np.array([0.4702203, 0.36157889, -0.80507986]),
        ]

        super().__init__(site_coords)


class SquareAntiprismAtom(AtomType):
    def __init__(self):
        """
        Approximate square antiprism geometry by optimisation of [XeF8]2-
        """
        site_coords = [
            np.array([-0.12556124, 0.56801979, -0.81338053]),
            np.array([-0.9236697, -0.18379172, -0.33623635]),
            np.array([-0.13553555, -0.94631881, 0.29344647]),
            np.array([0.19928886, -0.57504162, -0.79348036]),
            np.array([0.54323185, -0.11585026, 0.83155148]),
            np.array([-0.62763462, 0.12547974, 0.76832911]),
            np.array([0.10138916, 0.94996354, 0.29544797]),
            np.array([0.95080191, 0.18740602, -0.24668749]),
        ]

        super().__init__(site_coords)
