from autode.wrappers.keywords import Functional

pbe0 = Functional(name='pbe0',
                  doi_list=['10.1063/1.478522', '10.1103/PhysRevLett.77.3865'],
                  orca='PBE0',
                  g09='PBE1PBE',
                  nwchem='pbe0')

pbe = Functional(name='pbe',
                 doi_list=['10.1103/PhysRevLett.77.3865'],
                 orca='PBE',
                 g09='PBEPBE',
                 nwchem='xpbe96 cpbe96')
