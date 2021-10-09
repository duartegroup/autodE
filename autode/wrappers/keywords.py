from typing import Union
from copy import deepcopy
from abc import ABC, abstractmethod
from autode.log import logger


class KeywordsSet:

    def __init__(self, low_opt=None, grad=None, opt=None, opt_ts=None,
                 hess=None, optts_block='', sp=None, ecp=None):
        """
        Keywords used to specify the type and method used in electronic
        structure theory calculations. The input file for a single point
        calculation will look something like::

            -------------------------------------------------------------------
            <keyword line directive> autode.Keywords.x[0] ...
            autode.Keywords.optts_block

            <coordinate directive> <charge> <multiplicity>
            .
            .
            coordinates
            .
            .
            <end of coordinate directive>
            -------------------------------------------------------------------

        Keyword Arguments:

            low_opt (list(str)): List of keywords for a low level optimisation

            grad (list(str)): List of keywords for a gradient calculation

            opt (list(str)): List of keywords for a low level optimisation

            opt_ts (list(str)): List of keywords for a low level optimisation

            hess (list(str)): List of keywords for a low level optimisation

            optts_block (str): String as extra input for a TS optimisation

            sp  (list(str)): List of keywords for a single point calculation

            ecp (autode.wrrappers.keywords.ECP | None): Effective core
                potential for atoms heavier than
        """

        self.low_opt = OptKeywords(low_opt)
        self.opt = OptKeywords(opt)
        self.opt_ts = OptKeywords(opt_ts)

        self.grad = GradientKeywords(grad)
        self.hess = HessianKeywords(hess)

        self.low_sp = None                      # Low level single point
        self.sp = SinglePointKeywords(sp)

        self._list = [self.low_opt, self.opt, self.opt_ts, self.grad,
                      self.hess, self.sp]

        self.optts_block = optts_block

        if ecp is not None:
            self.set_ecp(ecp)

    def __repr__(self):
        str_methods = ',\n'.join(str(c) for c in self._list if c is not None)
        return f'KeywordsSet({str_methods})'

    def __getitem__(self, item):
        return self._list[item]

    def set_opt_functional(self, functional):
        """Set the functional for all optimisation and gradient calculations"""
        for attr in ('low_opt', 'opt', 'opt_ts', 'grad', 'hess'):
            getattr(self, attr).functional = functional

        return None

    def set_opt_basis_set(self, basis_set):
        """Set the basis set for all optimisation and gradient calculations"""
        for attr in ('low_opt', 'opt', 'opt_ts', 'grad', 'hess'):
            getattr(self, attr).basis_set = basis_set

        return None

    def set_functional(self, functional):
        """Set the functional for all calculation types"""
        for keywords in self:
            keywords.functional = functional

        return None

    def set_dispersion(self, dispersion):
        """Set the dispersion correction for all calculation types"""
        for keywords in self:
            keywords.dispersion = dispersion

        return None

    def set_ecp(self, ecp):
        """Set the effective core potential for all calculation types"""
        for keywords in self:
            keywords.ecp = ecp

        return None


class Keywords:

    def __init__(self, keyword_list=None):
        """
        Read only list of keywords

        Keyword Arguments:
            keyword_list (list(str)): List of keywords used in a QM calculation
        """
        self.keyword_list = keyword_list if keyword_list is not None else []

    def __str__(self):
        return '_'.join([str(kw) for kw in self.keyword_list])

    def __repr__(self):
        """Representation of these keywords"""
        return self.__str__()

    def _string(self, prefix):
        """Return a string defining the keywords, with or without a prefix"""
        from autode.config import Config
        base_str = ' '.join([str(kw) for kw in self.keyword_list])

        return f'{prefix}({base_str})'

    def _get_keyword(self, keyword_type):
        """Get a keyword given a type"""

        for keyword in self.keyword_list:
            if isinstance(keyword, keyword_type):
                return keyword

        return None

    def _set_keyword(self, keyword, keyword_type):
        """Set a keyword. A keyword of the same type must exist"""
        if type(keyword) is str:
            keyword = keyword_type(name=keyword)

        assert type(keyword) is keyword_type or keyword is None

        for i, keyword_in_list in enumerate(self.keyword_list):
            if isinstance(keyword_in_list, keyword_type):
                if keyword is None:
                    del self.keyword_list[i]
                else:
                    self.keyword_list[i] = keyword
                return

            # Cannot have both wavefunction and DFT methoda
            if ((isinstance(keyword_in_list, WFMethod)
                 and keyword_type == Functional)
                or
                (isinstance(keyword_in_list, Functional)
                 and keyword_type == WFMethod)):

                raise ValueError('Could not set a functional with a '
                                 'WF method present, or vice-versa ')

        # This keyword does not appear in the list, so add it
        self.append(keyword)
        return None

    @property
    def ecp(self):
        """Get the effective core potential used"""
        return self._get_keyword(ECP)

    @ecp.setter
    def ecp(self, ecp):
        """Set the functional in a set of keywords"""
        self._set_keyword(ecp, keyword_type=ECP)

    @property
    def functional(self):
        """Get the functional in this set of keywords"""
        return self._get_keyword(Functional)

    @property
    def basis_set(self):
        """Get the functional in this set of keywords"""
        return self._get_keyword(BasisSet)

    @property
    def dispersion(self):
        """Get the dispersion keyword in this set of keywords"""
        return self._get_keyword(DispersionCorrection)

    @property
    def wf_method(self):
        """Get the wavefunction method in this set of keywords"""
        return self._get_keyword(WFMethod)

    @wf_method.setter
    def wf_method(self, method):
        self._set_keyword(method, keyword_type=WFMethod)

    @functional.setter
    def functional(self, functional):
        """Set the functional in a set of keywords"""
        self._set_keyword(functional, keyword_type=Functional)

    @dispersion.setter
    def dispersion(self, dispersion):
        """Set the dispersion correction in a set of keywords"""
        self._set_keyword(dispersion, keyword_type=DispersionCorrection)

    @basis_set.setter
    def basis_set(self, basis_set):
        """Set the functional in a set of keywords"""
        self._set_keyword(basis_set, keyword_type=BasisSet)

    @property
    def method_string(self):
        """Generate a string with refs (dois) for this method e.g. PBE0-D3BJ"""
        string = ''

        func = self.functional
        if func is not None:
            string += f'{func.upper()}({func.doi_str})'

        disp = self.dispersion
        if disp is not None:
            string += f'-{disp.upper()}({disp.doi_str})'

        wf = self.wf_method
        if wf is not None:
            string += f'{str(wf)}({wf.doi_str})'

        ri = self._get_keyword(keyword_type=RI)
        if ri is not None:
            string += f'({ri.upper()}, {ri.doi_str})'

        if len(string) == 0:
            logger.warning('Unknown method')
            string = '???'

        return string

    @property
    def bstring(self):
        """Brief string without dois of the method e.g. PBE0-D3BJ/def2-SVP"""

        string = ''

        if self.functional is not None:
            string += self.functional.upper()

        if self.wf_method is not None:
            string += f'-{self.wf_method.upper()}'

        if self.dispersion is not None:
            string += f'-{self.dispersion.upper()}'

        if self.basis_set is not None:
            string += f'/{self.basis_set.name}'

        return string

    def copy(self):
        return deepcopy(self)

    def append(self, item):
        assert type(item) is str or isinstance(item, Keyword)

        # Don't re-add a keyword that is already there
        if any(kw.lower() == item.lower() for kw in self.keyword_list):
            return

        self.keyword_list.append(item)

    def remove(self, item):
        self.keyword_list.remove(item)

    def __getitem__(self, item):
        return self.keyword_list[item]

    def __len__(self):
        return len(self.keyword_list)

    def __iter__(self):
        return iter(self.keyword_list)


class OptKeywords(Keywords):

    @property
    def max_opt_cycles(self):
        """
        Maximum number of optimisation cycles

        Returns:
            (autode.wrappers.keywords.MaxOptCycles):
        """
        return self._get_keyword(MaxOptCycles)

    @max_opt_cycles.setter
    def max_opt_cycles(self, value: Union[int, 'MaxOptCycles', None]):
        """Set the maximum number of optimisation cycles"""
        if value is None:
            self._set_keyword(None, MaxOptCycles)
            return

        if int(value) <= 0:
            raise ValueError('Must have a positive number of opt cycles')

        self._set_keyword(MaxOptCycles(int(value)), MaxOptCycles)

    def __str__(self):
        return self._string(prefix='OptKeywords')


class HessianKeywords(Keywords):

    def __str__(self):
        return self._string(prefix='HessKeywords')


class GradientKeywords(Keywords):

    def __str__(self):
        return self._string(prefix='GradKeywords')


class SinglePointKeywords(Keywords):

    def __str__(self):
        return self._string(prefix='SPKeywords')


class Keyword(ABC):

    def __init__(self, name, doi=None, doi_list=None, **kwargs):
        """
        A keyword for an electronic structure theory method e.g. basis set or
        functional, with possibly a an associated reference or set of
        references.

        e.g.
        keyword = Keyword(name='pbe')
        keyword = Keyword(name='pbe', g09='xpbe96 cpbe96')

        ---------------------------------------------------------------------
        Arguments:
            name: (str) Name of the keyword/method
            doi: (str) Digital object identifier for the method's paper

        Keyword Arguments:
            kwargs: (str) Keyword in a particular electronic structure theory
                          package e.g. Keyword(..., orca='PBE0') for a
                          functional
        """
        self.name = name

        self.doi_list = []
        if doi is not None:
            self.doi_list.append(doi)

        if doi_list is not None:
            self.doi_list += doi_list

        # Update the attributes with any keyword arguments
        self.__dict__.update(kwargs)

        # Gaussian 09 and Gaussian 16 keywords are the same(?)
        if 'g09' in kwargs.keys():
            self.g16 = kwargs['g09']

    @abstractmethod
    def __repr__(self):
        """Representation of this keyword"""

    def __eq__(self, other):
        return str(self) == str(other)

    def __str__(self):
        return self.name

    def lower(self):
        return self.name.lower()

    def upper(self):
        return self.name.upper()

    @property
    def doi_str(self):
        return ' '.join(self.doi_list)

    def has_only_name(self):
        """Determine if only a name has been set, in which case it will
        be printed verbatim into an input file, otherwise needs keyword.method
        to be set, where method is e.g. orca"""

        excl_attrs = ('name', 'doi_list')
        for attr in self.__dict__:
            if attr in excl_attrs:
                continue
            return False

        return True


class BasisSet(Keyword):
    """Basis set for a QM method"""

    def __repr__(self):
        return f'BasisSet({self.name})'


class DispersionCorrection(Keyword):
    """Functional for a DFT method"""

    def __repr__(self):
        return f'DispersionCorrection({self.name})'


class Functional(Keyword):
    """Functional for a DFT method"""

    def __repr__(self):
        return f'Functional({self.name})'


class ImplicitSolventType(Keyword):
    """
    A type of implicit solvent model. Example::

        cpcm = ImplicitSolventType(name='cpcm', doi='10.the_doi')
    """

    def __repr__(self):
        return f'ImplicitSolventType({self.name})'


class RI(Keyword):
    """Resolution of identity approximation"""

    def __repr__(self):
        return f'ResolutionOfIdentity({self.name})'


class WFMethod(Keyword):
    """Keyword for a wavefunction method e.g. HF or CCSD(T)"""

    def __repr__(self):
        return f'WaveFunctionMethod({self.name})'


class ECP(Keyword):
    """Effective core potential"""

    def __repr__(self):
        return f'EffectiveCorePotential({self.name})'

    def __eq__(self, other):
        """Equality of ECPs"""
        return (isinstance(other, ECP)
                and str(self) == str(other)
                and self.min_atomic_number == other.min_atomic_number)

    def __init__(self, name, min_atomic_number=37,
                 doi=None, doi_list=None, **kwargs):
        """
        An effective core potential that applies to all atoms with atomic
        numbers larger than min_atomic_number

        Arguments:
            name (str):
            min_atomic_number (int):
            doi (str):
            doi_list (list(str)):
            kwargs:
        """
        super().__init__(name, doi, doi_list, **kwargs)

        self.min_atomic_number = min_atomic_number


class MaxOptCycles(Keyword):
    """Maximum number of optimisation cycles"""

    def __repr__(self):
        return f'MaxOptCycles(N = {self.name})'

    def __int__(self):
        return int(self.name)

    def __init__(self,
                 number: int):
        super().__init__(name=str(int(number)))
