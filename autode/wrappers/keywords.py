from copy import deepcopy
from autode.log import logger


class KeywordsSet:

    def __getitem__(self, item):
        return self._list[item]

    def set_opt_functional(self, functional):
        """Set the functional for all optimisation and gradient calculations"""
        for attr in ('opt', 'opt_ts', 'grad', 'hess'):
            getattr(self, attr).set_functional(functional)

        return None

    def set_opt_basis_set(self, basis_set):
        """Set the basis set for all optimisation and gradient calculations"""
        for attr in ('opt', 'opt_ts', 'grad', 'hess'):
            getattr(self, attr).set_basis_set(basis_set)

        return None

    def set_functional(self, functional):
        """Set the functional for all calculation types"""
        assert type(functional) is Functional

        for keywords in self:
            keywords.set_functional(functional)

        return None

    def set_dispersion(self, dispersion):
        """Set the dispersion correction for all calculation types"""
        assert type(dispersion) is DispersionCorrection or dispersion is None

        for keywords in self:
            keywords.set_dispersion(dispersion)

        return None

    def __init__(self, low_opt=None, grad=None, opt=None, opt_ts=None,
                 hess=None, optts_block='', sp=None):
        """
        Keywords used to specify the type and method used in electronic
        structure theory calculations. The input file for a single point
        calculation will look something like:

        ---------------------------------------------------------------------
        <keyword line directive> autode.Keywords.sp[0] autode.Keywords.sp[1]
        autode.Keywords.optts_block

        <coordinate directive> <charge> <multiplicity>
        .
        .
        coordinates
        .
        .
        <end of coordinate directive>
        ---------------------------------------------------------------------
        Keyword Arguments:

            low_opt (list(str)): List of keywords for a low level optimisation
            grad (list(str)): List of keywords for a gradient calculation
            opt (list(str)): List of keywords for a low level optimisation
            opt_ts (list(str)): List of keywords for a low level optimisation
            hess (list(str)): List of keywords for a low level optimisation
            optts_block (str): String as extra input for a TS optimisation
            sp  (list(str)): List of keywords for a single point calculation
        :return:
        """

        self.low_opt = OptKeywords(low_opt)
        self.opt = OptKeywords(opt)
        self.opt_ts = OptKeywords(opt_ts)

        self.grad = GradientKeywords(grad)
        self.hess = HessianKeywords(hess)

        self.sp = SinglePointKeywords(sp)

        self._list = [self.low_opt, self.opt, self.opt_ts, self.grad,
                      self.hess, self.sp]

        self.optts_block = optts_block


class Keywords:

    def __str__(self):
        return '_'.join([str(kw) for kw in self.keyword_list])

    def _string(self, prefix):
        """Return a string defining the keywords, with or without a prefix"""
        from autode.config import Config
        base_str = '_'.join([str(kw) for kw in self.keyword_list])

        if Config.keyword_prefixes:
            return f'{prefix}({base_str})'
        else:
            return base_str

    def _set_keyword(self, keyword, keyword_type):
        """Set a keyword. A keyword of the same type must exist"""
        if type(keyword) is str:
            keyword = Functional(name=keyword)

        assert type(keyword) is keyword_type or keyword is None

        for i, keyword_in_list in enumerate(self.keyword_list):
            if isinstance(keyword_in_list, keyword_type):
                if keyword is None:
                    self.keyword_list.pop(i)
                else:
                    self.keyword_list[i] = keyword
                return

        raise ValueError('Could not set the keyword - none recognised')

    def set_functional(self, functional):
        """Set the functional in a set of keywords"""
        return self._set_keyword(functional, keyword_type=Functional)

    def set_dispersion(self, dispersion):
        """Set the dispersion correction in a set of keywords"""
        return self._set_keyword(dispersion, keyword_type=DispersionCorrection)

    def set_basis_set(self, basis_set):
        """Set the functional in a set of keywords"""
        return self._set_keyword(basis_set, keyword_type=BasisSet)

    def _get_keyword(self, keyword_type):
        """Get a keyword given a type"""

        for keyword in self.keyword_list:
            if isinstance(keyword, keyword_type):
                return keyword

        return None

    def method_string(self):
        """Generate a string with refs (dois) for this method e.g. PBE0-D3BJ"""
        string = ''

        func = self.functional()
        if func is not None:
            string += f'{func.upper()}({func.doi_str()})'

        disp = self.dispersion()
        if disp is not None:
            string += f'-{disp.upper()}({disp.doi_str()})'

        wf = self.wf_method()
        if wf is not None:
            string += f'{str(wf)}({wf.doi_str()})'

        ri = self._get_keyword(keyword_type=RI)
        if ri is not None:
            string += f'({ri.upper()}, {ri.doi_str()})'

        if len(string) == 0:
            logger.warning('Unknown method')
            string = '???'

        return string

    def functional(self):
        """Get the functional in this set of keywords"""
        return self._get_keyword(Functional)

    def basis_set(self):
        """Get the functional in this set of keywords"""
        return self._get_keyword(BasisSet)

    def dispersion(self):
        """Get the dispersion keyword in this set of keywords"""
        return self._get_keyword(DispersionCorrection)

    def wf_method(self):
        """Get the wavefunction method in this set of keywords"""
        return self._get_keyword(WFMethod)

    def copy(self):
        return deepcopy(self.keyword_list)

    def append(self, item):
        assert type(item) is str

        # Don't re-add a keyword that is already there
        if any(kw.lower() == item.lower() for kw in self.keyword_list):
            return

        self.keyword_list.append(item)

    def remove(self, item):
        self.keyword_list.remove(item)

    def __getitem__(self, item):
        return self.keyword_list[item]

    def __init__(self, keyword_list):
        """
        Read only list of keywords

        Args:
            keyword_list (list(str)): List of keywords used in a QM calculation
        """
        self.keyword_list = keyword_list if keyword_list is not None else []


class OptKeywords(Keywords):

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


class Keyword:

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name

    def lower(self):
        return self.name.lower()

    def upper(self):
        return self.name.upper()

    def doi_str(self):
        return ' '.join(self.doi_list)

    def __init__(self, name, doi=None, doi_list=None, **kwargs):
        """
        A keyword for an electronic structure theory method e.g. basis set or
        functional, with possibly a an associated reference or set of
        references

        ---------------------------------------------------------------------
        Arguments:
            name: (str) Name of the model
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


class BasisSet(Keyword):
    """Basis set for a QM method"""


class DispersionCorrection(Keyword):
    """Functional for a DFT method"""


class Functional(Keyword):
    """Functional for a DFT method"""


class ImplicitSolventType(Keyword):
    """
    A type of implicit solvent model. Example::

        cpcm = ImplicitSolventType(name='cpcm', doi='10.the_doi')
    """


class RI(Keyword):
    """Resolution of identity approximation"""


class WFMethod(Keyword):
    """Keyword for a wavefunction method e.g. HF or CCSD(T)"""
