from typing import Optional
from autode.exceptions import AtomsNotFound
from autode.log import logger
from autode.species.species import Species


class Conformer(Species):

    def __init__(self,
                 name:         str = 'conf',
                 atoms:        Optional['autode.atoms.Atoms'] = None,
                 solvent_name: Optional[str] = None,
                 charge:       int = 0,
                 mult:         int = 1,
                 dist_consts:  Optional[dict] = None,
                 species:      Optional[Species] = None):
        """
        Construct a conformer either using the standard species constructor,
        or from a species directly.

        -----------------------------------------------------------------------
        See Also:
            (autode.species.species.Species):
        """

        super().__init__(name, atoms, charge, mult, solvent_name=solvent_name)

        if species is not None:
            self.charge = species.charge  # Require identical charge/mult/solv
            self.mult = species.mult
            self.solvent = species.solvent

            if atoms is None:
                self.atoms = species.atoms.copy()

        self.constraints.update(distance=dist_consts)

    def __repr__(self):
        """Representation of """
        return self._repr(prefix='Conformer')

    def __eq__(self, other):
        return super().__eq__(other)

    def single_point(self,
                     method:  'autode.wrappers.base.ElectronicStructureMethod',
                     keywords: Optional['autode.wrappers.keywords.Keywords'] = None,
                     n_cores:  Optional[int] = None):
        """
        Calculate a single point and default to a low level single point method

        ----------------------------------------------------------------------
        Arguments:
            method (autode.wrappers.base.ElectronicStructureMethod):

            keywords (autode.wrappers.keywords.Keywords):

            n_cores (int | None): If None then defaults to Config.n_cores
        """
        keywords = method.keywords.low_sp if keywords is None else keywords

        return super().single_point(method, keywords, n_cores=n_cores)

    def optimise(self,
                 method:      Optional['autode.wrappers.base.ElectronicStructureMethod'] = None,
                 reset_graph: bool = False,
                 calc:        Optional['autode.calculation.Calculation'] = None,
                 keywords:    Optional['autode.wrappers.keywords.Keywords'] = None,
                 n_cores:     Optional[int] = None):
        """
        Optimise the geometry of this conformer using a method. Will use
        low_opt keywords if no keywords are given.

        -----------------------------------------------------------------------
        Arguments:
            method (autode.wrappers.base.ElectronicStructureMethod):

            reset_graph (bool):

            calc (autode.calculation.Calculation):

            keywords (autode.wrappers.keywords.Keywords):

            n_cores (int | None): If None then defaults to Config.n_cores
        """
        try:
            if keywords is None and method is not None:
                keywords = method.keywords.low_opt

            super().optimise(method, keywords=keywords, calc=calc, n_cores=n_cores)

        except AtomsNotFound:
            logger.error(f'Atoms not found for {self.name} but not critical')
            self.atoms = None

        return None
