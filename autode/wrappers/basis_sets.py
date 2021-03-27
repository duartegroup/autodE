from autode.wrappers.keywords import BasisSet, ECP

def2svp = BasisSet(name='def2-SVP', doi='10.1039/B508541A',
                   orca='def2-SVP',
                   g09='Def2SVP',
                   nwchem='Def2-SVP')

def2tzvp = BasisSet(name='def2-TZVP', doi='10.1039/B508541A',
                    orca='def2-TZVP',
                    g09='Def2TZVP',
                    nwchem='Def2-TZVP')


def2ecp = ECP(name='def2-ECP',
              doi_list=['Ce-Yb: 10.1063/1.456066',
                        'Y-Cd, Hf-Hg: 10.1007/BF01114537',
                        'Te-Xe, In-Sb, Ti-Bi: 10.1063/1.1305880',
                        'Po-Rn: 10.1063/1.1622924',
                        'Rb, Cs: 10.1016/0009-2614(96)00382-X',
                        'Sr, Ba: 10.1063/1.459993',
                        'La: 10.1007/BF00528565',
                        'Lu: 10.1063/1.1406535'],
              orca=None,              # def2-ECP is applied by default
              nwchem='def2-ecp',
              g09='',  # TODO: Gaussian implementation
              min_atomic_number=37)   # applies to Rb and heavier
