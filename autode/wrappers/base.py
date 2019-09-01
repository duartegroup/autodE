from autode.config import Config
from shutil import which
from functools import wraps


def add_method(cls):
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            return func(*args, **kwargs)
        setattr(cls, func.__name__, wrapper)
        # Note we are not binding func, but wrapper which accepts self but does exactly the same as func
        return func # returning func means func can still be used normally
    return decorator


class ElectronicStructureMethod:

    def __init__(self, name, path, req_licence=False, path_to_licence=None, aval_solvents=None):

        self.path = path if path is not None else which(name)      # If the path is not set in config.py search in $PATH
        self.aval_solvents = aval_solvents

        if req_licence:
            self.available = True if path is not None and path_to_licence is not None else False
        else:
            self.available = True if path is not None else False


smd_solvents = ['1,1,1-TRICHLOROETHANE', 'CYCLOPENTANE', '1,1,2-TRICHLOROETHANE', 'CYCLOPENTANOL',
                '1,2,4-TRIMETHYLBENZENE', 'CYCLOPENTANONE', '1,2-DIBROMOETHANE', '1,2-DICHLOROETHANE ',
                'CIS-DECALIN',
                '1,2-ETHANEDIOL ', 'N-DECANE', '1,4-DIOXANEDIBROMOMETHANE', '1-BROMO-2-METHYLPROPANE',
                'DIBUTYLETHER',
                '1-BROMOOCTANE', 'O-DICHLOROBENZENE', '1-BROMOPENTANE', 'E-1,2-DICHLOROETHENE', '1-BROMOPROPANE ',
                'Z-1,2-DICHLOROETHENE', '1-BUTANOL', 'DICHLOROMETHANE', '1-CHLOROHEXANE', '1-CHLOROPENTANE',
                '1-CHLOROPROPANE', 'DIETHYLAMINE', '1-DECANOL', 'DIIODOMETHANE', '1-FLUOROOCTANE', '1-HEPTANOL',
                'CIS-1,2-DIMETHYLCYCLOHEXANE', 'DECALIN', 'DIETHYL ETHER ', 'DIETHYL SULFIDE', 'DIISOPROPYL ETHER',
                '1-HEXANOL', '1-HEXENE', 'N,N-DIMETHYLACETAMIDE', '1-HEXYNE', 'N,N-DIMETHYLFORMAMIDE DMF',
                '1-IODOBUTANE', 'DIMETHYLSULFOXIDE DMSO', '1-IODOHEXADECANE', 'DIPHENYLETHER', '1-IODOPENTANE',
                'DIPROPYLAMINE', '1-IODOPROPANE', 'N-DODECANE', '1-NITROPROPANE', 'ETHANETHIOL', '1-NONANOL',
                'ETHANOL',
                '1-OCTANOL', '1-PENTANOL', '1-PENTENE', '1-PROPANOL', 'ETHYLBENZENE', '2,2,2-TRIFLUOROETHANOL',
                'LUOROBENZENE', '2,2,4-TRIMETHYLPENTANE', 'FORMAMIDE', '2,4-DIMETHYLPENTANE',
                '2,4-DIMETHYLPYRIDINE',
                'N-HEPTANE', '2,6-DIMETHYLPYRIDINE', 'N-HEXADECANE', '2-BROMOPROPANE', 'N-HEXANE',
                'DIMETHYL DISULFIDE',
                'ETHYL ETHANOATE ', 'ETHYL METHANOATE ', 'ETHYL PHENYL ETHER', 'FORMIC ACID', '2-BUTANOL',
                'HEXANOIC ACID', '2-CHLOROBUTANE', '2-HEPTANONE', '2-HEXANONE', '2-METHOXYETHANOL',
                '2-METHYL-1-PROPANOL', '2-METHYL-2-PROPANOL', '2-METHYLPENTANE', '2-METHYLPYRIDINE',
                '2-NITROPROPANE', '2-OCTANONE', '2-PENTANONE', 'IODOBENZENE', 'IODOETHANE', 'IODOMETHANE',
                'ISOPROPYLBENZENE', 'P-ISOPROPYLTOLUENE', 'MESITYLENE', 'METHANOL', 'METHYL BENZOATE',
                'METHYL BUTANOATE', 'METHYL ETHANOATE', 'METHYL METHANOATE', 'METHYL PROPANOATE', 'N-METHYLANILINE',
                'METHYLCYCLOHEXANE', 'N-METHYLFORMAMIDE (E/Z MIXTURE)', 'NITROBENZENE', 'PhNO2', 'NITROETHANE',
                'NITROMETHANE', 'MeNO2 ', 'O-NITROTOLUENE', 'N-NONANE', 'N-OCTANE', 'N-PENTADECANE', 'PENTANAL',
                'N-PENTANE', 'PENTANOIC ACID', 'PENTYL ETHANOATE', 'PENTYLAMINE', 'PERFLUOROBENZENE', 'PROPANAL',
                'PROPANOIC ACID', 'PROPANONITRILE', 'PROPYL ETHANOATE', 'PROPYLAMINE', 'PYRIDINE',
                'TETRACHLOROETHENE',
                'TETRAHYDROFURAN', 'THF', 'TETRAHYDROTHIOPHENE-S,S-DIOXIDE', 'TETRALIN', 'THIOPHENE', 'THIOPHENOL',
                'TOLUENE', 'TRANS-DECALIN', 'TRIBUTYLPHOSPHATE', 'TRICHLOROETHENE', 'TRIETHYLAMINE', 'N-UNDECANE',
                'WATER', 'XYLENE (MIXTURE)', 'M-XYLENE', 'O-XYLENE', 'P-XYLENE', '2-PROPANOL', '2-PROPEN-1-OL',
                'E-2-PENTENE', '3-METHYLPYRIDINE', '3-PENTANONE', '4-HEPTANONE', '4-METHYL-2-PENTANONE',
                '4-METHYLPYRIDINE', '5-NONANONE', 'ACETIC ACID', 'ACETONE', 'ACETONITRILE MeCN', 'ACETOPHENONE',
                'ANILINE', 'ANISOLE', 'BENZALDEHYDE', 'BENZENE', 'BENZONITRILE', 'BENZYL ALCOHOL', 'BROMOBENZENE',
                'BROMOETHANE', 'BROMOFORM', 'BUTANAL', 'BUTANOIC ACID', 'BUTANONE', 'BUTANONITRILE',
                'BUTYL ETHANOATE',
                'BUTYLAMINE', 'N-BUTYLBENZENE', 'SEC-BUTYLBENZENE', 'TERT-BUTYLBENZENE', 'CARBON DISULFIDE',
                'CARBON TETRACHLORIDE', 'CHLOROBENZENE', 'CHLOROFORM', 'A-CHLOROTOLUENE', 'O-CHLOROTOLUENE',
                'M-CRESOL',
                'O-CRESOL', 'CYCLOHEXANE', 'CYCLOHEXANONE']

xtb_solvents = ['Acetone', 'Acetonitrile', 'Benzene', 'CH2Cl2', 'CHCl3', 'CS2', 'DMF', 'DMSO', 'Ether', 'Water',
                'H2O', 'Methanol', 'n-Hexane', 'THF', 'Toluene']

ORCA = ElectronicStructureMethod(name='orca',
                                 path=Config.ORCA.path,
                                 aval_solvents=[solv.lower() for solv in smd_solvents])

XTB = ElectronicStructureMethod(name='xtb',
                                path=Config.XTB.path,
                                aval_solvents=[solv.lower() for solv in xtb_solvents])

MOPAC = ElectronicStructureMethod(name='mopac',
                                  path=Config.MOPAC.path,
                                  req_licence=True,
                                  path_to_licence=Config.MOPAC.licence_path)

PSI4 = ElectronicStructureMethod(name='psi4',
                                 path=Config.PSI4.path,
                                 aval_solvents=None)
