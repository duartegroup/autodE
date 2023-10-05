from typing import Union, Optional, Sequence, List, Type, Iterator, TypeVar
from copy import deepcopy
from abc import ABC, abstractmethod
from autode.log import logger

TypeKeywords = TypeVar("TypeKeywords", bound="Keywords")


class KeywordsSet:
    def __init__(
        self,
        low_opt: Optional["_KEYWORDS_TYPE"] = None,
        grad: Optional["_KEYWORDS_TYPE"] = None,
        low_sp: Optional["_KEYWORDS_TYPE"] = None,
        opt: Optional["_KEYWORDS_TYPE"] = None,
        opt_ts: Optional["_KEYWORDS_TYPE"] = None,
        hess: Optional["_KEYWORDS_TYPE"] = None,
        sp: Optional["_KEYWORDS_TYPE"] = None,
        ecp: Optional["ECP"] = None,
    ):
        """
        Keywords used to specify the type and method used in electronic
        structure theory calculations. The input file for a single point
        calculation will look something like::

            -------------------------------------------------------------------
            <keyword line directive> autode.KeywordsSet.keywords[0] ...

            <coordinate directive> <charge> <multiplicity>
            .
            .
            coordinates
            .
            .
            <end of coordinate directive>
            -------------------------------------------------------------------

        -----------------------------------------------------------------------
        Arguments:

            low_opt: List of keywords for a low level optimisation

            grad: List of keywords for a gradient calculation

            low_sp: Low-level single point

            opt: List of keywords for a optimisation

            opt_ts: List of keywords for a transition state optimisation

            hess: List of keywords for a hessian calculation

            sp: List of keywords for a single point calculation

            ecp: Effective core potential to use for atoms heavier than
                 ecp.min_atomic_number, if not None

             optts_block: String as extra input for a TS optimisation
        """

        self._low_opt: OptKeywords = OptKeywords(low_opt)
        self._opt: OptKeywords = OptKeywords(opt)
        self._opt_ts: OptTSKeywords = OptTSKeywords(opt_ts)

        self._grad: GradientKeywords = GradientKeywords(grad)
        self._hess: HessianKeywords = HessianKeywords(hess)

        self._low_sp: SinglePointKeywords = SinglePointKeywords(low_sp)
        self._sp: SinglePointKeywords = SinglePointKeywords(sp)

        if ecp is not None:
            self.set_ecp(ecp)

    def __repr__(self):
        str_methods = ",\n".join(str(c) for c in self._list if c is not None)
        return f"KeywordsSet({str_methods})"

    def __getitem__(self, item: int) -> "Keywords":
        return self._list[item]

    def __iter__(self) -> Iterator["Keywords"]:
        yield from self._list

    def __eq__(self, other: object) -> bool:
        """Equality of two keyword sets"""
        return isinstance(other, KeywordsSet) and self._list == other._list

    @property
    def low_opt(self) -> "OptKeywords":
        return self._low_opt

    @low_opt.setter
    def low_opt(self, value: Optional[Sequence[str]]):
        self._low_opt = OptKeywords(value)

    @property
    def opt(self) -> "OptKeywords":
        return self._opt

    @opt.setter
    def opt(self, value: Optional[Sequence[str]]):
        self._opt = OptKeywords(value)

    @property
    def opt_ts(self) -> "OptTSKeywords":
        return self._opt_ts

    @opt_ts.setter
    def opt_ts(self, value: Optional[Sequence[str]]):
        self._opt_ts = OptTSKeywords(value)

    @property
    def grad(self) -> "GradientKeywords":
        return self._grad

    @grad.setter
    def grad(self, value: Optional[Sequence[str]]):
        self._grad = GradientKeywords(value)

    @property
    def hess(self) -> "HessianKeywords":
        return self._hess

    @hess.setter
    def hess(self, value: Optional[Sequence[str]]):
        self._hess = HessianKeywords(value)

    @property
    def low_sp(self) -> "SinglePointKeywords":
        return self._low_sp

    @low_sp.setter
    def low_sp(self, value: Optional[Sequence[str]]):
        self._low_sp = SinglePointKeywords(value)

    @property
    def sp(self) -> "SinglePointKeywords":
        return self._sp

    @sp.setter
    def sp(self, value: Optional[Sequence[str]]):
        self._sp = SinglePointKeywords(value)

    @property
    def _list(self) -> List["Keywords"]:
        """List of all the keywords in this set"""
        return [
            self._low_opt,
            self._opt,
            self._opt_ts,
            self._grad,
            self._hess,
            self._sp,
            self._low_sp,
        ]

    def set_opt_functional(self, functional: Union["Functional", str]):
        """Set the functional for all optimisation and gradient calculations"""
        for attr in ("low_opt", "opt", "opt_ts", "grad", "hess"):
            getattr(self, attr).functional = functional

        return None

    def set_opt_basis_set(self, basis_set: Union["BasisSet", str]):
        """Set the basis set for all optimisation and gradient calculations"""
        for attr in ("low_opt", "opt", "opt_ts", "grad", "hess"):
            getattr(self, attr).basis_set = basis_set

        return None

    def set_functional(self, functional: Union["Functional", str]):
        """Set the functional for all calculation types"""
        for keywords in self:
            keywords.functional = functional

        return None

    def set_dispersion(self, dispersion: Union["DispersionCorrection", str]):
        """Set the dispersion correction for all calculation types"""
        for keywords in self:
            keywords.dispersion = dispersion

        return None

    def set_ecp(self, ecp: Union["ECP", str]):
        """Set the effective core potential for all calculation types"""
        for keywords in self:
            keywords.ecp = ecp

        return None

    def copy(self) -> "KeywordsSet":
        return deepcopy(self)


class Keywords(ABC):
    def __init__(
        self, keyword_list: Union["_KEYWORDS_TYPE", str, None] = None
    ):
        """
        List of keywords used in an electronic structure calculation

        -----------------------------------------------------------------------
        Arguments:
            keyword_list: Keywords
        """

        self._list: List[Union[Keyword, str]] = []

        if isinstance(keyword_list, str):
            self._list = [keyword_list]
        elif keyword_list is not None:
            self._list = list(keyword_list)

    def __str__(self):
        return " ".join([repr(kw) for kw in self._list])

    def __eq__(self, other: object) -> bool:
        """Equality of these keywords to another kind"""
        return isinstance(other, self.__class__) and set(self._list) == set(
            other._list
        )

    def __add__(self, other: object):
        """Add some keywords to these"""

        if isinstance(other, Keywords):
            return self.__class__(self._list + other._list)

        elif isinstance(other, list):
            return self.__class__(self._list + other)

        else:
            raise ValueError(
                f"Cannot add {other} to the keywords. Must be a "
                f"list or a Keywords object"
            )

    @abstractmethod
    def __repr__(self):
        """Representation of these keywords"""

    def _get_keyword(
        self, keyword_type: Type["Keyword"]
    ) -> Optional["Keyword"]:
        """Get a keyword given a type"""

        for keyword in self._list:
            if isinstance(keyword, keyword_type):
                return keyword

        return None

    def _set_keyword(
        self,
        keyword: Union["Keyword", str, None],
        keyword_type: Type["Keyword"],
    ):
        """Set a keyword. A keyword of the same type must exist"""
        if type(keyword) is str:
            keyword = keyword_type(name=keyword)

        assert type(keyword) is keyword_type or keyword is None

        for i, keyword_in_list in enumerate(self._list):
            if isinstance(keyword_in_list, keyword_type):
                if keyword is None:
                    del self._list[i]
                else:
                    self._list[i] = keyword
                return

            # Cannot have both wavefunction and DFT methoda
            if (
                isinstance(keyword_in_list, WFMethod)
                and keyword_type == Functional
            ) or (
                isinstance(keyword_in_list, Functional)
                and keyword_type == WFMethod
            ):
                raise ValueError(
                    "Could not set a functional with a "
                    "WF method present, or vice-versa "
                )

        if keyword is None:  # don't append None to list
            return

        # This keyword does not appear in the list, so add it
        self.append(keyword)
        return None

    def tolist(self) -> List:
        return self._list

    @property
    def ecp(self):
        """Get the effective core potential used"""
        return self._get_keyword(ECP)

    @ecp.setter
    def ecp(self, ecp: Union["ECP", str]):
        """Set the functional in a set of keywords"""
        self._set_keyword(ecp, keyword_type=ECP)

    @property
    def functional(self):
        """Get the functional in this set of keywords"""
        return self._get_keyword(Functional)

    @functional.setter
    def functional(self, functional: Union["Functional", str]):
        """Set the functional in a set of keywords"""
        self._set_keyword(functional, keyword_type=Functional)

    @property
    def basis_set(self):
        """Get the functional in this set of keywords"""
        return self._get_keyword(BasisSet)

    @basis_set.setter
    def basis_set(self, basis_set: Union["BasisSet", str]):
        """Set the functional in a set of keywords"""
        self._set_keyword(basis_set, keyword_type=BasisSet)

    @property
    def dispersion(self):
        """Get the dispersion keyword in this set of keywords"""
        return self._get_keyword(DispersionCorrection)

    @dispersion.setter
    def dispersion(self, dispersion: Union["DispersionCorrection", str]):
        """Set the dispersion correction in a set of keywords"""
        self._set_keyword(dispersion, keyword_type=DispersionCorrection)

    @property
    def wf_method(self):
        """Get the wavefunction method in this set of keywords"""
        return self._get_keyword(WFMethod)

    @wf_method.setter
    def wf_method(self, method: Union["WFMethod", str]):
        self._set_keyword(method, keyword_type=WFMethod)

    @property
    def method_string(self) -> str:
        """Generate a string with refs (dois) for this method e.g. PBE0-D3BJ"""
        string = ""

        func = self.functional
        if func is not None:
            string += f"{func.upper()}({func.doi_str})"

        disp = self.dispersion
        if disp is not None:
            string += f"-{disp.upper()}({disp.doi_str})"

        wf = self.wf_method
        if wf is not None:
            string += f"{str(wf)}({wf.doi_str})"

        ri = self._get_keyword(keyword_type=RI)
        if ri is not None:
            string += f"({ri.upper()}, {ri.doi_str})"

        if len(string) == 0:
            logger.warning("Unknown method")
            string = "???"

        return string

    @property
    def bstring(self) -> str:
        """Brief string without dois of the method e.g. PBE0-D3BJ/def2-SVP"""

        string = ""

        if self.functional is not None:
            string += self.functional.upper()

        if self.wf_method is not None:
            string += f"-{self.wf_method.upper()}"

        if self.dispersion is not None:
            string += f"-{self.dispersion.upper()}"

        if self.basis_set is not None:
            string += f"/{self.basis_set.name}"

        return string

    def contain_any_of(self, *words: str) -> bool:
        """
        Do these keywords contain any of a set of other words? Not case
        sensitive.

        -----------------------------------------------------------------------
        Arguments:
            *words: Words that may be present in these keywords

        Returns:
            (bool):
        """
        kwds = set(w.lower() for w in self)

        return not kwds.isdisjoint(w.lower() for w in words)

    def copy(self) -> TypeKeywords:  # type: ignore
        return deepcopy(self)  # type: ignore

    def append(self, item: Union["Keyword", str]) -> None:
        assert type(item) is str or isinstance(item, Keyword)

        # Don't re-add a keyword that is already there
        if any(kw.lower() == item.lower() for kw in self._list):
            return

        self._list.append(item)

    def remove(self, item: "Keyword") -> None:
        self._list.remove(item)

    def __getitem__(self, item: int) -> Union["Keyword", str]:
        return self._list[item]

    def __setitem__(self, key: int, value: Union["Keyword", str]) -> None:
        self._list[key] = value

    def __len__(self) -> int:
        return len(self._list)

    def __iter__(self) -> Iterator:
        return iter(self._list)


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
    def max_opt_cycles(self, value: Union[int, "MaxOptCycles", None]):
        """Set the maximum number of optimisation cycles"""
        if value is None:
            self._set_keyword(None, MaxOptCycles)
            return

        if int(value) <= 0:
            raise ValueError("Must have a positive number of opt cycles")

        self._set_keyword(MaxOptCycles(int(value)), MaxOptCycles)

    def __repr__(self):
        return f"OptKeywords({self.__str__()})"


class OptTSKeywords(OptKeywords):
    """Transition state optimisation keywords"""


class HessianKeywords(Keywords):
    def __repr__(self):
        return f"HessKeywords({self.__str__()})"


class GradientKeywords(Keywords):
    def __repr__(self):
        return f"GradKeywords({self.__str__()})"


class SinglePointKeywords(Keywords):
    def __repr__(self):
        return f"SPKeywords({self.__str__()})"


class Keyword(ABC):
    def __init__(
        self, name: str, doi_list: Optional[List[str]] = None, **kwargs
    ):
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

        self.g09: Optional[str] = None
        self.g16: Optional[str] = None
        self.qchem: Optional[str] = None
        self.orca: Optional[str] = None
        self.xtb: Optional[str] = None
        self.nwchem: Optional[str] = None

        self.doi_list = []
        if "doi" in kwargs and kwargs["doi"] is not None:
            self.doi_list.append(kwargs.pop("doi"))

        if doi_list is not None:
            self.doi_list += doi_list

        # Update the attributes with any keyword arguments
        self.__dict__.update(kwargs)

        # Gaussian 09 and Gaussian 16 keywords are the same
        if "g09" in kwargs.keys():
            self.g16 = kwargs["g09"]

    @abstractmethod
    def __repr__(self):
        """Representation of this keyword"""

    def __eq__(self, other):
        return str(self) == str(other)

    def __str__(self):
        return self.name

    def __hash__(self):
        """Unique hash of this object"""
        return hash(repr(self))

    def lower(self):
        return self.name.lower()

    def upper(self):
        return self.name.upper()

    @property
    def doi_str(self):
        return " ".join(self.doi_list)

    @property
    def has_only_name(self):
        """
        Determine if only a name has been set, in which case it will
        be printed verbatim into an input file, otherwise needs keyword.method
        to be set, where method is e.g. orca
        """
        excl = ("name", "doi_list", "doi", "freq_scale_factor")
        return all(
            getattr(self, a) is None for a in self.__dict__ if a not in excl
        )


class BasisSet(Keyword):
    """Basis set for a QM method"""

    def __repr__(self):
        return f"BasisSet({self.name})"


class DispersionCorrection(Keyword):
    """Functional for a DFT method"""

    def __repr__(self):
        return f"DispersionCorrection({self.name})"


class Functional(Keyword):
    """Functional for a DFT method"""

    def __init__(
        self,
        name,
        doi=None,
        doi_list=None,
        freq_scale_factor: float = 1.0,
        **kwargs,
    ):
        super().__init__(name, doi=doi, doi_list=doi_list, **kwargs)

        self.freq_scale_factor = freq_scale_factor

    def __repr__(self):
        return f"Functional({self.name})"

    def __eq__(self, other):
        return isinstance(other, Functional) and self.name == other.name

    def __hash__(self):
        return hash(self.name)


class ImplicitSolventType(Keyword):
    """
    A type of implicit solvent model. Example::

        cpcm = ImplicitSolventType(name='cpcm', doi='10.the_doi')
    """

    def __repr__(self):
        return f"ImplicitSolventType({self.name})"


class RI(Keyword):
    """Resolution of identity approximation"""

    def __repr__(self):
        return f"ResolutionOfIdentity({self.name})"


class WFMethod(Keyword):
    """Keyword for a wavefunction method e.g. HF or CCSD(T)"""

    def __repr__(self):
        return f"WaveFunctionMethod({self.name})"


class ECP(Keyword):
    """Effective core potential"""

    def __repr__(self):
        return f"EffectiveCorePotential({self.name})"

    def __eq__(self, other: object):
        """Equality of ECPs"""
        return (
            isinstance(other, ECP)
            and str(self) == str(other)
            and self.min_atomic_number == other.min_atomic_number
        )

    def __hash__(self):
        """Unique hash of this effective core potential"""
        return hash(str(self) + str(self.min_atomic_number))

    def __init__(
        self,
        name: str,
        min_atomic_number: int = 37,
        doi: Optional[str] = None,
        doi_list: Optional[List[str]] = None,
        **kwargs,
    ):
        """
        An effective core potential that applies to all atoms with atomic
        numbers larger than min_atomic_number

        -----------------------------------------------------------------------
        Arguments:
            name (str):
            min_atomic_number (int):
            doi (str):
            doi_list (list(str)):
            kwargs:
        """
        super().__init__(name, doi=doi, doi_list=doi_list, **kwargs)

        self.min_atomic_number = min_atomic_number


class MaxOptCycles(Keyword):
    """Maximum number of optimisation cycles"""

    def __repr__(self):
        return f"MaxOptCycles(N = {self.name})"

    def __int__(self):
        return int(self.name)

    def __init__(self, number: int):
        super().__init__(name=str(int(number)))


_KEYWORDS_TYPE = Union[Keywords, Sequence[Union[Keyword, str]]]
