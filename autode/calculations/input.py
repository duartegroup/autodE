import os
import autode.wrappers.keywords as kws

from typing import Optional, List, TYPE_CHECKING
from autode.point_charges import PointCharge

if TYPE_CHECKING:
    from autode.wrappers.keywords import Keywords


class CalculationInput:
    def __init__(
        self,
        keywords: "Keywords",
        added_internals: Optional[list] = None,
        point_charges: Optional[List[PointCharge]] = None,
    ):
        """
        Calculation input

        -----------------------------------------------------------------------
        Arguments:
            keywords: Keywords that a method will use to run the calculation
                      e.g. ['pbe', 'def2-svp'] for an ORCA single point at
                      PBE/def2-SVP

            added_internals: Atom indexes to add to the internal coordinates

            point_charges: Optional list of float of point charges, x, y, z
                           coordinates for each point charge
        """
        self.keywords: Keywords = keywords.copy()

        self.added_internals: Optional[list] = None
        if added_internals is not None and len(added_internals) > 0:
            self.added_internals = added_internals

        self.point_charges = point_charges

        self.filename: Optional[str] = None
        self.additional_filenames: List[str] = []

        self._check()

    def _check(self):
        """Check that the input parameters have the expected format"""
        if self.keywords is not None:
            assert isinstance(self.keywords, kws.Keywords)

        # Ensure the point charges are given as a list of PointCharge objects
        if self.point_charges is not None:
            assert type(self.point_charges) is list
            assert all(type(pc) is PointCharge for pc in self.point_charges)

        if self.added_internals is not None:
            assert type(self.added_internals) is list
            assert all(len(idxs) == 2 for idxs in self.added_internals)

    @property
    def exists(self):
        """Does the input (files) exist?"""
        return self.filename is not None and all(
            os.path.exists(fn) for fn in self.filenames
        )

    @property
    def filenames(self):
        """Return a list of all the input files"""
        if self.filename is None:
            return self.additional_filenames

        return [self.filename] + self.additional_filenames
