"""
Functional instances. Frequency scale factors have been obtained from
https://cccbdb.nist.gov/vibscalejust.asp and the basis set dependence assumed
to be negligible at least double zetas (i.e. >6-31G)
"""
from autode.wrappers.keywords.keywords import Functional

pbe0 = Functional(
    name="pbe0",
    doi_list=["10.1063/1.478522", "10.1103/PhysRevLett.77.3865"],
    orca="PBE0",
    g09="PBE1PBE",
    nwchem="pbe0",
    qchem="pbe0",
    freq_scale_factor=0.96,
)

pbe = Functional(
    name="pbe",
    doi_list=["10.1103/PhysRevLett.77.3865"],
    orca="PBE",
    g09="PBEPBE",
    nwchem="xpbe96 cpbe96",
    qchem="pbe",
    freq_scale_factor=0.99,
)
