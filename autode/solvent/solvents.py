from autode.exceptions import SolventNotFound


def get_solvent(solvent_name, implicit=True, explicit=False):
    """For a named solvent_name return the autode.solvent.solvents.Solvent
    that is one of the aliases """
    if solvent_name is None:
        return None

    if implicit and not explicit:

        for solvent in solvents:
            if solvent_name.lower() in solvent.aliases:
                return solvent

        raise SolventNotFound('No matching solvent in the library')

    if explicit:
        # Return ExplicitSolvent
        raise NotImplementedError

    return None


def get_available_solvent_names(method):
    """For an autode.wrappers.base.ElectronicStructureMethod return those
    solvents that are available"""
    return [s.name for s in solvents if hasattr(s, method.name)]


class Solvent:

    def __str__(self):
        return self.name

    def __eq__(self, other):
        """Determine if two solvent are the same based on name and SMILES"""
        if other is None:
            return False

        return self.name == other.name and self.smiles == other.smiles

    def __init__(self, name, smiles, aliases, **kwargs):
        """
        Solvent class. As electronic structure methods implement implicit
        solvation without a unique list of solvents there needs to be
        conversion between them, while also allowing for user specifying
        one possibility from a list of aliases

        Arguments:
            name (str): Unique name of the solvent
            smiles (str): SMILES string
            aliases (list(str)): Different names for the same solvent e.g.
                                 water and H2O

        Keyword Arguments:
            kwargs (str): Name of the solvent in the electronic structure
                          package e.g. Solvent(..., orca='water')
        """

        self.name = name
        self.smiles = smiles
        self.aliases = aliases

        # Add attributes for all the methods specified e.g. initialisation with
        # orca='water' -> self.orca = 'water'
        self.__dict__.update(kwargs)

        # Gaussian 09 and Gaussian 16 solvents are named the same
        if 'g09' in kwargs.keys():
            self.g16 = kwargs['g09']


class ExplicitSolvent(Solvent):
    # Implement explicit solvation here
    pass


solvents = [Solvent(name='water', smiles='O', aliases=['water', 'h2o'], orca='water', g09='Water', nwchem='water', xtb='Water', mopac='water'),
            Solvent(name='dichloromethane', smiles='ClCCl', aliases=['dichloromethane', 'methyl dichloride', 'dcm'], orca='dichloromethane', g09='Dichloromethane', nwchem='dcm', xtb='CH2Cl2', mopac='dichloromethane'),
            Solvent(name='acetone', smiles='CC(C)=O', aliases=['acetone', 'propanone'], orca='acetone', g09='Acetone', nwchem='acetone', xtb='Acetone', mopac='acetone'),
            Solvent(name='acetonitrile', smiles='CC#N', aliases=['acetonitrile', 'mecn', 'ch3cn'], orca='acetonitrile', g09='Acetonitrile', nwchem='acetntrl', xtb='Acetonitrile', mopac='acetonitrile'),
            Solvent(name='benzene', smiles='C1=CC=CC=C1', aliases=['benzene', 'cyclohexatriene'], orca='benzene', g09='Benzene', nwchem='benzene', xtb='Benzene', mopac='benzene'),
            Solvent(name='trichloromethane', smiles='ClC(Cl)Cl', aliases=['chloroform', 'trichloromethane', 'chcl3', 'methyl trichloride'], orca='chloroform', g09='Chloroform', nwchem='chcl3', xtb='CHCl3', mopac='chloroform'),
            Solvent(name='cs2', smiles='S=C=S', aliases=['cs2', 'methanedithione', 'carbon bisulfide'], orca='carbon disulfide', g09='CarbonDiSulfide', nwchem='cs2', xtb='CS2', mopac='cs2'),
            Solvent(name='dmf', smiles='O=CN(C)C', aliases=['dmf', 'dimethylformamide', 'n,n-dimethylformamide'], orca='n,n-dimethylformamide', g09='n,n-DiMethylFormamide', nwchem='dmf', xtb='DMF', mopac='n,n-dimethylformamide'),
            Solvent(name='dmso', smiles='O=S(C)C', aliases=['dmso', 'dimethylsulfoxide'], orca='dimethylsulfoxide', g09='DiMethylSulfoxide', nwchem='dmso', xtb='DMSO', mopac='dmso'),
            Solvent(name='diethyl ether', smiles='CCOCC', aliases=['diethyl ether', 'ether', 'Ethoxyethane'], orca='diethyl ether', g09='DiethylEther', nwchem='ether', xtb='Ether', mopac='ether'),
            Solvent(name='methanol', smiles='CO', aliases=['methanol', 'meoh'], orca='methanol', g09='Methanol', nwchem='methanol', xtb='Methanol', mopac='methanol'),
            Solvent(name='hexane', smiles='CCCCCC', aliases=['hexane', 'n-hexane'], orca='n-hexane', g09='n-Hexane', nwchem='hexane', xtb='n-Hexane', mopac='hexane'),
            Solvent(name='thf', smiles='C1CCOC1', aliases=['thf', 'tetrahydrofuran', 'oxolane'], orca='tetrahydrofuran', g09='TetraHydroFuran', nwchem='thf', xtb='THF', mopac='tetrahydrofuran'),
            Solvent(name='toluene', smiles='CC1=CC=CC=C1', aliases=['toluene', 'methylbenzene', 'phenyl methane'], orca='toluene', g09='Toluene', nwchem='toluene', xtb='Toluene', mopac='toluene'),
            Solvent(name='acetic acid', smiles='CC(O)=O', aliases=['acetic acid', 'ethanoic acid'], orca='acetic acid', g09='AceticAcid', nwchem='acetacid', mopac='acetic acid'),
            Solvent(name='1-butanol', smiles='CCCCO', aliases=['1-butanol', 'butanol', 'n-butanol', 'butan-1-ol'], orca='1-butanol', g09='1-Butanol', nwchem='butanol', mopac='1-butanol'),
            Solvent(name='2-butanol', smiles='CC(O)CC', aliases=['2-butanol', 'sec-butanol', 'butan-2-ol'], orca='2-butanol', g09='2-Butanol', nwchem='butanol2', mopac='2-butanol'),
            Solvent(name='acetophenone', smiles='CC(C1=CC=CC=C1)=O', aliases=['acetophenone', 'phenylacetone', 'phenylethanone'], orca='acetophenone', g09='AcetoPhenone', nwchem='acetphen', mopac='acetophenone'),
            Solvent(name='aniline', smiles='NC1=CC=CC=C1', aliases=['aniline', 'benzenamine', 'phenylamine'], orca='aniline', g09='Aniline', nwchem='aniline', mopac='aniline'),
            Solvent(name='anisole', smiles='COC1=CC=CC=C1', aliases=['anisole', 'methoxybenzene', 'phenoxymethane'], orca='anisole', g09='Anisole', nwchem='anisole', mopac='anisole'),
            Solvent(name='benzaldehyde', smiles='O=CC1=CC=CC=C1', aliases=['benzaldehyde', 'phenylmethanal'], orca='benzaldehyde', g09='Benzaldehyde', nwchem='benzaldh', mopac='benzaldehyde'),
            Solvent(name='benzonitrile', smiles='N#CC1=CC=CC=C1', aliases=['benzonitrile', 'cyanobenzene', 'phenyl cyanide'], orca='benzonitrile', g09='BenzoNitrile', nwchem='benzntrl', mopac='benzonitrile'),
            Solvent(name='benzyl chloride', smiles='ClCC1=CC=CC=C1', aliases=['benzyl chloride', '(chloromethyl)benzene', 'Chloromethyl benzene', 'a-chlorotoluene'], orca='a-chlorotoluene', g09='a-ChloroToluene', nwchem='benzylcl', mopac='benzyl chloride'),
            Solvent(name='1-bromo-2-methylpropane', smiles='CC(C)CBr', aliases=['1-bromo-2-methylpropane', 'isobutyl bromide'], orca='1-bromo-2-methylpropane', g09='1-Bromo-2-MethylPropane', nwchem='brisobut', mopac='isobutyl bromide'),
            Solvent(name='bromobenzene', smiles='BrC1=CC=CC=C1', aliases=['bromobenzene', 'phenyl bromide'], orca='bromobenzene', g09='BromoBenzene', nwchem='brbenzen', mopac='bromobenzene'),
            Solvent(name='bromoethane', smiles='CCBr', aliases=['bromoethane', 'ethyl bromide', 'etbr'], orca='bromoethane', g09='BromoEthane', nwchem='brethane', mopac='bromoethane'),
            Solvent(name='bromoform', smiles='BrC(Br)Br', aliases=['bromoform', 'tribromomethane', 'methyl tribromide', 'chbr3'], orca='bromoform', g09='Bromoform', nwchem='bromform', mopac='bromoform'),
            Solvent(name='1-bromooctane', smiles='CCCCCCCCBr', aliases=['1-bromooctane', 'bromooctane', 'octyl bromide', '1-octyl bromide'], orca='1-bromooctane', g09='1-BromoOctane', nwchem='broctane', mopac='bromooctane'),
            Solvent(name='1-bromopentane', smiles='CCCCCBr', aliases=['1-bromopentane', 'bromopentane', 'pentyl bromide'], orca='1-bromopentane', g09='1-BromoPentane', nwchem='brpentan', mopac='bromopentane'),
            Solvent(name='butantal', smiles='CCCC=O', aliases=['butanal', 'butyraldehyde'], orca='butanal', g09='Butanal', nwchem='butanal', mopac='butanal'),
            Solvent(name='butanone', smiles='CC(CC)=O', aliases=['butanone', '2-butanone', 'butan-2-one', 'methyl ethyl ketone', 'ethyl methyl ketone'], orca='butanone', g09='Butanone', nwchem='butanone', mopac='2-butanone'),
            Solvent(name='carbon tetrachloride', smiles='ClC(Cl)(Cl)Cl', aliases=['carbon tetrachloride', 'ccl4', 'tetrachloromethane'], orca='carbon tetrachloride', g09='CarbonTetraChloride', nwchem='carbntet', mopac='carbon tetrachloride'),
            Solvent(name='chlorobenzene', smiles='ClC1=CC=CC=C1', aliases=['chlorobenzene', 'benzene chloride', 'phenyl chloride'], orca='chlorobenzene', g09='ChloroBenzene', nwchem='clbenzen', mopac='chlorobenzene'),
            Solvent(name='cyclohexane', smiles='C1CCCCC1', aliases=['cyclohexane'], orca='cyclohexane', g09='CycloHexane', nwchem='cychexan', mopac='cyclohexane'),
            Solvent(name='1,2-dichlorobenzene', smiles='ClC1=CC=CC=C1Cl', aliases=['1,2-dichlorobenzene', 'o-dichlorobenzene', 'ortho-dichlorobenzene'], orca='o-dichlorobenzene', g09='o-DiChloroBenzene', nwchem='odiclbnz', mopac='1,2-dichlorobenzene'),
            Solvent(name='n,n-dimethylacetamide', smiles='CC(N(C)C)=O', aliases=['n,n-dimethylacetamide', 'dmac', 'dma', 'dimethylacetamide'], orca='n,n-dimethylacetamide', g09='n,n-DiMethylAcetamide', nwchem='dma', mopac='n,n-dimethylacetamide'),
            Solvent(name='dioxane', smiles='O1CCOCC1', aliases=['dioxane', '1,4-dioxane', 'p-dioxane'], orca='1,4-dioxane', g09='1,4-Dioxane', nwchem='dioxane', mopac='1,4-dioxane'),
            Solvent(name='ethyl acetate', smiles='CC(OCC)=O', aliases=['ethyl acetate', 'etoac', 'ethyl ethanoate'], orca='ethyl ethanoate', g09='EthylEthanoate', nwchem='etoac', mopac='ethyl acetate'),
            Solvent(name='ethanol', smiles='CCO', aliases=['ethanol', 'ethyl alcohol', 'etoh'], orca='ethanol', g09='Ethanol', nwchem='ethanol', mopac='ethyl alcohol'),
            Solvent(name='heptane', smiles='CCCCCCC', aliases=['heptane', 'n-heptane'], orca='n-heptane', g09='Heptane', nwchem='heptane', mopac='heptane'),
            Solvent(name='pentane', smiles='CCCCC', aliases=['pentane', 'n-pentane'], orca='n-pentane', g09='n-Pentane', nwchem='npentane', mopac='pentane'),
            Solvent(name='1-propanol', smiles='CCCO', aliases=['1-propanol', 'propanol', 'n-propaol', 'n-proh'], orca='1-propanol', g09='1-Propanol', nwchem='propanol', mopac='1-propanol'),
            Solvent(name='pyridine', smiles='C1=NC=CC=C1', aliases=['pyridine'], orca='pyridine', g09='Pyridine', nwchem='pyridine', mopac='pyridine'),
            Solvent(name='1,1,1-trichloroethane', smiles='CC(Cl)(Cl)Cl', aliases=['1,1,1-trichloroethane', 'methyl chloroform', '1,1,1-tca'], orca='1,1,1-trichloroethane', g09='1,1,1-TriChloroEthane', nwchem='tca111', mopac='1,1,1-trichloroethane'),
            Solvent(name='cyclopentane', smiles='C1CCCC1', aliases=['cyclopentane'], orca='cyclopentane', g09='CycloPentane', nwchem='cycpentn', mopac='cyclopentane'),
            Solvent(name='1,1,2-trichloroethane', smiles='ClCC(Cl)Cl', aliases=['1,1,2-trichloroethane', 'vinyl trichloride', '1,1,2-tca'], orca='1,1,2-trichloroethane', g09='1,1,2-TriChloroEthane', nwchem='tca112', mopac='1,1,2-trichloroethane'),
            Solvent(name='cyclopentanol', smiles='OC1CCCC1', aliases=['cyclopentanol'], orca='cyclopentanol', g09='CycloPentanol', nwchem='cycpntol', mopac='cyclopentanol'),
            Solvent(name='1,2,4-trimethylbenzene', smiles='CC1=CC=C(C)C(C)=C1', aliases=['1,2,4-trimethylbenzene', 'pseudocumene'], orca='1,2,4-trimethylbenzene', g09='1,2,4-TriMethylBenzene', nwchem='tmben124', mopac='1,2,4-trimethylbenzene'),
            Solvent(name='cyclopentanone', smiles='O=C1CCCC1', aliases=['cyclopentanone'], orca='cyclopentanone', g09='CycloPentanone', nwchem='cycpnton', mopac='cyclopentanone'),
            Solvent(name='1,2-dibromoethane', smiles='BrCCBr', aliases=['1,2-dibromoethane', 'ethylene dibromide', 'edb'], orca='1,2-dibromoethane', g09='1,2-DiBromoEthane', nwchem='edb12', mopac='1,2-dibromoethane'),
            Solvent(name='1,2-dichloroethane', smiles='ClCCCl', aliases=['1,2-dichloroethane', 'ethylene dichloride', 'dce', 'dichloroethane'], orca='1,2-dichloroethane', g09='DiChloroEthane', nwchem='edc12', mopac='1,2-dichloroethane'),
            Solvent(name='cis-decalin', smiles='[H][C@@]12CCCC[C@]1([H])CCCC2', aliases=['cis-decalin', 'cis decalin'], orca='cis-decalin', g09='Cis-Decalin', nwchem='declncis', mopac='cis-decalin'),
            Solvent(name='trans-decalin', smiles='[H][C@@]12CCCC[C@@]1([H])CCCC2', aliases=['trans-decalin', 'trans decalin'], orca='trans-decalin', g09='trans-Decalin', nwchem='declntra', mopac='trans-decalin'),
            Solvent(name='decalin mix', smiles='C12CCCCC1CCCC2', aliases=['decalin mix', 'decalin', 'decalin mixture'], orca='decalin', g09='Decalin-mixture', nwchem='declnmix', mopac='decalin'),
            Solvent(name='1,2-ethanediol', smiles='OCCO', aliases=['1,2-ethanediol', 'ethylene glycol', 'ethane-1,2-diol', 'monoethylene glycol'], orca='1,2-ethanediol', g09='1,2-EthaneDiol', nwchem='meg', mopac='1,2-ethanediol'),
            Solvent(name='decane', smiles='CCCCCCCCCC', aliases=['decane', 'n-decane'], orca='n-decane', g09='n-Decane', nwchem='decane', mopac='decane'),
            Solvent(name='dibromomethane', smiles='BrCBr', aliases=['dibromomethane', 'methyl dibromide'], orca='dibromomethane', g09='DiBromomEthane', nwchem='dibrmetn', mopac='dibromomethane'),
            Solvent(name='dibutylether', smiles='CCCCOCCCC', aliases=['dibutylether', 'butyl ether'], orca='dibutylether', g09='DiButylEther', nwchem='butyleth', mopac='dibutylether'),
            Solvent(name='cis-1,2-dichloroethene', smiles='Cl/C=C\\Cl', aliases=['cis-1,2-dichloroethene', 'cis-1,2-dichloroethylene', 'z-1,2-dichloroethene','z-1,2-dichloroethylene'], orca='z-1,2-dichloroethene', g09='z-1,2-DiChloroEthene', nwchem='c12dce', mopac='z-1,2-dichloroethene'),
            Solvent(name='trans-1,2-dichloroethen', smiles='Cl/C=C/Cl', aliases=['trans-1,2-dichloroethene', 'trans-1,2-dichloroethylene', 'e-1,2-dichloroethene', 'e-1,2-dichloroethylene'], orca='e-1,2-dichloroethene', g09='e-1,2-DiChloroEthene', nwchem='t12dce', mopac='z-1,2-dichloroethene'),
            Solvent(name='1-bromopropane', smiles='CCCBr', aliases=['1-bromopropane', 'bromopropane'], orca='1-bromopropane', g09='1-BromoPropane', nwchem='brpropan', mopac='1-bromopropane'),
            Solvent(name='2-bromopropane', smiles='CC(Br)C', aliases=['2-bromopropane', 'isopropyl bromide'], orca='2-bromopropane', g09='2-BromoPropane', nwchem='brpropa2', mopac='2-bromopropane'),
            Solvent(name='1-chlorohexane', smiles='CCCCCCCl', aliases=['1-chlorohexane', 'chlorohexane'], orca='1-chlorohexane', g09='1-ChloroHexane', nwchem='clhexane', mopac='1-chlorohexane'),
            Solvent(name='1-chloropentane', smiles='CCCCCCl', aliases=['1-chloropentane', 'chloropentane'], orca='1-chloropentane', g09='1-ChloroPentane', nwchem='clpentan', mopac='1-chloropentane'),
            Solvent(name='1-chloropropane', smiles='CCCCl', aliases=['1-chloropropane', 'chloropropane'], orca='1-chloropropane', g09='1-ChloroPropane', nwchem='clpropan', mopac='1-chloropropane'),
            Solvent(name='diethylamine', smiles='CCNCC', aliases=['diethylamine', 'n-ethylethanamine'], orca='diethylamine', g09='DiEthylAmine', nwchem='dietamin', mopac='diethylamine'),
            Solvent(name='1-decanol', smiles='CCCCCCCCCCO', aliases=['1-decanol', 'decanol', 'decan-1-ol'], orca='1-decanol', g09='1-Decanol', nwchem='decanol', mopac='decanol'),
            Solvent(name='diiodomethane', smiles='ICI', aliases=['diiodomethane', 'methylene iodide'], orca='diiodomethane', g09='DiIodoMethane', nwchem='mi', mopac='diiodomethane'),
            Solvent(name='1-fluorooctane', smiles='CCCCCCCCF', aliases=['1-fluorooctane', 'fluorooctane', 'octyl fluoride'], orca='1-fluorooctane', g09='1-FluoroOctane', nwchem='foctane', mopac='1-fluorooctane'),
            Solvent(name='1-heptanol', smiles='CCCCCCCO', aliases=['1-helptanol', 'heptanol', 'heptan-1-ol'], orca='1-helptanol', g09='1-Heptanol', nwchem='heptanol', mopac='heptanol'),
            Solvent(name='cis-1,2-dimethylcyclohexane', smiles='C[C@@H]1[C@H](C)CCCC1', aliases=['cis-1,2-dimethylcyclohexane'], orca='cis-1,2-dimethylcyclohexane', g09='Cis-1,2-DiMethylCycloHexane', nwchem='cisdmchx', mopac='cisdmchx'),
            Solvent(name='diethyl sulfide', smiles='CCSCC', aliases=['diethyl sulfide', 'et2s', 'thioethyl ether'], orca='diethyl sulfide', g09='DiEthylSulfide', nwchem='et2s', mopac='diethyl sulfide'),
            Solvent(name='diisopropyl ether', smiles='CC(OC(C)C)C', aliases=['diisopropyl ether', 'dipe'], orca='diisopropyl ether', g09='DiIsoPropylEther', nwchem='dipe', mopac='diisopropyl ether'),
            Solvent(name='1-hexanol', smiles='CCCCCCO', aliases=['1-hexanol', 'hexanol', 'haxan-1-ol'], orca='1-hexanol', g09='1-Hexanol', nwchem='hexanol', mopac='hexanol'),
            Solvent(name='1-hexene', smiles='C=CCCCC', aliases=['1-hexene', 'hexene', 'hex-1-ene'], orca='1-hexene', g09='1-Hexene', nwchem='hexene', mopac='hexene'),
            Solvent(name='1-hexyne', smiles='C#CCCCC', aliases=['1-hexyne', 'hexyne', 'hex-1-yne'], orca='1-hexyne', g09='1-Hexyne', nwchem='hexyne', mopac='hexyne'),
            Solvent(name='1-iodobutane', smiles='CCCCI', aliases=['1-iodobutane', 'iodobutane'], orca='1-iodobutane', g09='1-IodoButane', nwchem='iobutane', mopac='iodobutane'),
            Solvent(name='1-iodohexadecane', smiles='CCCCCCCCCCCCCCCCI', aliases=['1-iodohexadecane', 'iodohexadecane'], orca='1-iodohexadecane', g09='1-IodoHexaDecane', nwchem='iohexdec', mopac='1-iodohexadecane'),
            Solvent(name='diphenylether', smiles='C1(OC2=CC=CC=C2)=CC=CC=C1', aliases=['diphenylether', 'phenoxybenzene'], orca='diphenylether', g09='DiPhenylEther', nwchem='phoph', mopac='diphenylether'),
            Solvent(name='1-iodopentane', smiles='CCCCCI', aliases=['1-iodopentane', 'iodopentane'], orca='1-iodopentane', g09='1-IodoPentane', nwchem='iopentan', mopac='1-iodopentane'),
            Solvent(name='1-iodopropane', smiles='CCCI', aliases=['1-iodopropane', 'iodopropane'], orca='1-iodopropane', g09='1-IodoPropane', nwchem='iopropan', mopac='1-iodopropane'),
            Solvent(name='dipropylamine', smiles='CCCNCCC', aliases=['dipropylamine'], orca='dipropylamine', g09='DiPropylAmine', nwchem='dproamin', mopac='dipropylamine'),
            Solvent(name='n-dodecane', smiles='CCCCCCCCCCCC', aliases=['n-dodecane', 'dodecane'], orca='n-dodecane', g09='n-Dodecane', nwchem='dodecan', mopac='dodecane'),
            Solvent(name='1-nitropropane', smiles='CCC[N+]([O-])=O', aliases=['1-nitropropane'], orca='1-nitropropane', g09='1-NitroPropane', nwchem='ntrprop1', mopac='1-nitropropane'),
            Solvent(name='ethanethiol', smiles='CCS', aliases=['ethanethiol', 'ethane thiol', 'etsh'], orca='ethanethiol', g09='EthaneThiol', nwchem='etsh', mopac='ethanethiol'),
            Solvent(name='1-nonanol', smiles='CCCCCCCCCO', aliases=['1-nonanol', 'nonanol', 'nonan-1-ol'], orca='1-nonanol', g09='1-Nonanol', nwchem='nonanol', mopac='nonanol'),
            Solvent(name='1-octanol', smiles='CCCCCCCCO', aliases=['1-octanol', 'octanol', 'octan-1-ol'], orca='1-octanol', g09='n-Octanol', nwchem='octanol', mopac='octanol'),
            Solvent(name='1-pentanol', smiles='CCCCCO', aliases=['1-pentanol', 'pentanol', 'pentan-1-ol'], orca='1-pentanol', g09='1-Pentanol', nwchem='pentanol', mopac='pentanol'),
            Solvent(name='1-pentene', smiles='C=CCCC', aliases=['1-pentene', 'pentene', 'pent-1-ene'], orca='1-pentene', g09='1-Pentene', nwchem='pentene', mopac='pentene'),
            Solvent(name='ethyl benzene', smiles='CCC1=CC=CC=C1', aliases=['ethyl benzene', 'ethylbenzene', 'phenylethane'], orca='ethylbenzene', g09='EthylBenzene', nwchem='eb', mopac='ethylbenzene'),
            Solvent(name='2,2,2-trifluoroethanol', smiles='FC(F)(F)CO', aliases=['2,2,2-trifluoroethanol'], orca='2,2,2-trifluoroethanol', g09='2,2,2-TriFluoroEthanol', nwchem='tfe222', mopac='2,2,2-trifluoroethanol'),
            Solvent(name='fluorobenzene', smiles='FC1=CC=CC=C1', aliases=['fluorobenzene', 'phenyl fluoride', 'c6h5f'], orca='fluorobenzene', g09='FluoroBenzene', nwchem='c6h5f', mopac='fluorobenzene'),
            Solvent(name='2,2,4-trimethylpentane', smiles='CC(C)(C)CC(C)C', aliases=['2,2,4-trimethylpentane', 'isooctane'], orca='2,2,4-trimethylpentane', g09='2,2,4-TriMethylPentane', nwchem='isoctane', mopac='2,2,4-trimethylpentane'),
            Solvent(name='formamide', smiles='O=CN', aliases=['formamide'], orca='formamide', g09='Formamide', nwchem='formamid', mopac='formamide'),
            Solvent(name='2,4-dimethylpentane', smiles='CC(C)CC(C)C', aliases=['2,4-dimethylpentane', 'diisopropylmethane'], orca='2,4-dimethylpentane', g09='2,4-DiMethylPentane', nwchem='dmepen24', mopac='2,4-dimethylpentane'),
            Solvent(name='2,4-dimethylpyridine', smiles='CC1=CC(C)=NC=C1', aliases=['2,4-dimethylpyridine', '2,4-lutidine'], orca='2,4-dimethylpyridine', g09='2,4-DiMethylPyridine', nwchem='dmepyr24', mopac='2,4-dimethylpyridine'),
            Solvent(name='2,6-dimethylpyridine', smiles='CC1=CC=CC(C)=N1', aliases=['2,6-dimethylpyridine', '2,6-lutidine', 'lutidine'], orca='2,6-dimethylpyridine', g09='2,6-DiMethylPyridine', nwchem='dmepyr26', mopac='2,6-dimethylpyridine'),
            Solvent(name='n-hexadecane', smiles='CCCCCCCCCCCCCCCC', aliases=['n-hexadecane', 'hexadecane'], orca='n-hexadecane', g09='n-Hexadecane', nwchem='hexadecn', mopac='hexadecane'),
            Solvent(name='dimethyl disulfide', smiles='CSSC', aliases=['dimethyl disulfide', 'dmds', 'methyl disulfide'], orca='dimethyl disulfide', g09='DiMethylDiSulfide', nwchem='dmds', mopac='dimethyl disulfide'),
            Solvent(name='ethyl methanoate', smiles='O=COCC', aliases=['ethyl methanoate', 'ethyl formate', 'etome'], orca='ethyl methanoate', g09='EthylMethanoate', nwchem='etome', mopac='ethyl methanoate'),
            Solvent(name='ethyl phenyl ether', smiles='CCOC1=CC=CC=C1', aliases=['ethyl phenyl ether', 'phenetole', 'ethoxybenzene'], orca='ethyl phenyl ether', g09='EthylPhenylEther', nwchem='phentol', mopac='phenetole'),
            Solvent(name='formic acid', smiles='O=CO', aliases=['formic acid', 'methanoic acid'], orca='formic acid', g09='FormicAcid', nwchem='formacid', mopac='formic acid'),
            Solvent(name='hexanoic acid', smiles='CCCCCC(O)=O', aliases=['hexanoic acid', 'caproic acid'], orca='hexanoic acid', g09='HexanoicAcid', nwchem='hexnacid', mopac='hexanoic acid'),
            Solvent(name='2-chlorobutane', smiles='CC(Cl)CC', aliases=['2-chlorobutane', 'sec-butyl chloride'], orca='2-chlorobutane', g09='2-ChloroButane', nwchem='secbutcl', mopac='2-chlorobutane'),
            Solvent(name='2-heptanone', smiles='CC(CCCCC)=O', aliases=['2-heptanone', 'heptan-2-one'], orca='2-heptanone', g09='2-Heptanone', nwchem='heptnon2', mopac='2-heptanone'),
            Solvent(name='2-hexanone', smiles='CC(CCCC)=O', aliases=['2-hexanone', 'hexan-2-one'], orca='2-hexanone', g09='2-Hexanone', nwchem='hexanon2', mopac='2-hexanone'),
            Solvent(name='2-methoxyethanol', smiles='COCCO', aliases=['2-methoxyethanol', 'egme'], orca='2-methoxyethanol', g09='2-MethoxyEthanol', nwchem='egme', mopac='2-methoxyethanol'),
            Solvent(name='2-methyl-1-propanol', smiles='CC(C)CO', aliases=['2-methyl-1-propanol', 'isobutanol'], orca='2-methyl-1-propanol', g09='2-Methyl-1-Propanol', nwchem='isobutol', mopac='isobutanol'),
            Solvent(name='2-methyl-2-propanol', smiles='CC(O)(C)C', aliases=['2-methyl-2-propanol', 'tert-butanol'], orca='2-methyl-2-propanol', g09='2-Methyl-2-Propanol', nwchem='terbutol', mopac='tertbutanol'),
            Solvent(name='2-methylpentane', smiles='CC(C)CCC', aliases=['2-methylpentane', 'isohexane'], orca='2-methylpentane', g09='2-MethylPentane', nwchem='isohexan', mopac='2-methylpentane'),
            Solvent(name='2-methylpyridine', smiles='CC1=NC=CC=C1', aliases=['2-methylpyridine', '2-picoline'], orca='2-methylpyridine', g09='2-MethylPyridine', nwchem='mepyrid2', mopac='2-methylpyridine'),
            Solvent(name='2-nitropropane', smiles='CC([N+]([O-])=O)C', aliases=['2-nitropropane'], orca='2-nitropropane', g09='2-NitroPropane', nwchem='ntrprop2', mopac='2-nitropropane'),
            Solvent(name='2-octanone', smiles='CC(CCCCCC)=O', aliases=['2-octanone', 'octan-2-one'], orca='2-octanone', g09='2-Octanone', nwchem='octanon2', mopac='2-octanone'),
            Solvent(name='2-pentanone', smiles='CC(CCC)=O', aliases=['2-pentanone', 'pentan-2-one'], orca='2-pentanone', g09='2-Pentanone', nwchem='pentnon2', mopac='2-pentanone'),
            Solvent(name='iodobenzene', smiles='IC1=CC=CC=C1', aliases=['iodobenzene', 'phenyl iodide'], orca='iodobenzene', g09='IodoBenzene', nwchem='c6h5i', mopac='iodobenzene'),
            Solvent(name='iodoethane', smiles='CCI', aliases=['iodoethane', 'ethyl iodide'], orca='iodoethane', g09='IodoEthane', nwchem='c2h5i', mopac='iodoethane'),
            Solvent(name='iodomethane', smiles='CI', aliases=['iodomethane', 'methyl iodide', 'mei', 'ch3i'], orca='iodomethane', g09='IodoMethane', nwchem='ch3i', mopac='iodomethane'),
            Solvent(name='isopropylbenzene', smiles='CC(C1=CC=CC=C1)C', aliases=['isopropylbenzene', 'cumene'], orca='isopropylbenzene', g09='IsoPropylBenzene', nwchem='cumene', mopac='isopropylbenzene'),
            Solvent(name='p-isopropyltoluene', smiles='CC1=CC=C(C(C)C)C=C1', aliases=['p-isopropyltoluene', 'para-isopropyltoluene', 'p-cymene'], orca='p-isopropyltoluene', g09='p-IsoPropylToluene', nwchem='p-cymene', mopac='p-cymene'),
            Solvent(name='mesitylene', smiles='CC1=CC(C)=CC(C)=C1', aliases=['mesitylene'], orca='mesitylene', g09='Mesitylene', nwchem='mesityln', mopac='mesitylene'),
            Solvent(name='methyl benzoate', smiles='O=C(OC)C1=CC=CC=C1', aliases=['methyl benzoate'], orca='methyl benzoate', g09='MethylBenzoate', nwchem='mebnzate', mopac='methyl benzoate'),
            Solvent(name='methyl butanoate', smiles='CCCC(OC)=O', aliases=['methyl butanoate', 'methyl butyrate'], orca='methyl butanoate', g09='MethylButanoate', nwchem='mebutate', mopac='methyl butanoate'),
            Solvent(name='methyl ethanoate', smiles='CC(OC)=O', aliases=['methyl ethanoate', 'methyl acetate'], orca='methyl ethanoate', g09='MethylEthanoate', nwchem='meacetat', mopac='methyl acetate'),
            Solvent(name='methyl methanoate', smiles='O=COC', aliases=['methyl methanoate', 'methyl formate'], orca='methyl methanoate', g09='MethylMethanoate', nwchem='meformat', mopac='methyl formate'),
            Solvent(name='methyl propanoate', smiles='CCC(OC)=O', aliases=['methyl propanoate', 'methyl propionate'], orca='methyl propanoate', g09='MethylPropanoate', nwchem='mepropyl', mopac='methyl propanoate'),
            Solvent(name='n-methylaniline', smiles='CNC1=CC=CC=C1', aliases=['n-methylaniline', 'nma'], orca='n-methylaniline', g09='n-MethylAniline', nwchem='nmeaniln', mopac='n-methylaniline'),
            Solvent(name='methylcyclohexane', smiles='CC1CCCCC1', aliases=['methylcyclohexane'], orca='methylcyclohexane', g09='MethylCycloHexane', nwchem='mecychex', mopac='methylcyclohexane'),
            Solvent(name='n-methylformamide (e/z mixture)', smiles='O=CNC', aliases=['n-methylformamide', 'n-methylformamide (e/z mixture)', 'n-methylformamide mixture', 'n-methylformamide mix'], orca='n-methylformamide (e/z mixture)', g09='n-MethylFormamide-mixture', nwchem='nmfmixtr', mopac='nmfmixtr'),
            Solvent(name='nitrobenzene', smiles='O=[N+](C1=CC=CC=C1)[O-]', aliases=['nitrobenzene', 'phno2'], orca='nitrobenzene', g09='NitroBenzene', nwchem='c6h5no2', mopac='nitrobenzene'),
            Solvent(name='nitroethane', smiles='CC[N+]([O-])=O', aliases=['nitroethane', 'etno2'], orca='nitroethane', g09='NitroEthane', nwchem='c2h5no2', mopac='nitroethane'),
            Solvent(name='nitromethane', smiles='C[N+]([O-])=O', aliases=['nitromethane', 'meno2', 'ch3no2'], orca='nitromethane', g09='NitroMethane', nwchem='ch3no2', mopac='nitromethane'),
            Solvent(name='o-nitrotoluene', smiles='CC1=CC=CC=C1[N+]([O-])=O', aliases=['o-nitrotoluene', 'ortho-nitrotoluene'], orca='o-nitrotoluene', g09='o-NitroToluene', nwchem='ontrtolu', mopac='o-nitrotoluene'),
            Solvent(name='n-nonane', smiles='CCCCCCCCC', aliases=['n-nonane', 'nonane'], orca='n-nonane', g09='n-Nonane', nwchem='nonane', mopac='n-nonane'),
            Solvent(name='n-octane', smiles='CCCCCCCC', aliases=['n-octane', 'octane'], orca='n-octane', g09='n-Octane', nwchem='octane', mopac='n-octane'),
            Solvent(name='n-pentadecane', smiles='CCCCCCCCCCCCCCC', aliases=['n-pentadecane', 'pentadecane'], orca='n-pentadecane', g09='n-Pentadecane', nwchem='pentdecn', mopac='n-pentadecane'),
            Solvent(name='pentanal', smiles='CCCCC=O', aliases=['pentanal'], orca='pentanal', g09='Pentanal', nwchem='pentanal', mopac='pentanal'),
            Solvent(name='pentanoic acid', smiles='CCCCC(O)=O', aliases=['pentanoic acid', 'valeric acid'], orca='pentanoic acid', g09='PentanoicAcid', nwchem='pentacid', mopac='pentanoic acid'),
            Solvent(name='pentyl ethanoate', smiles='CC(OCCCCC)=O', aliases=['pentyl ethanoate', 'pentyl acetate'], orca='pentyl ethanoate', g09='PentylEthanoate', nwchem='pentacet', mopac='pentyl acetate'),
            Solvent(name='pentyl amine', smiles='NCCCCC', aliases=['pentyl amine', 'pentylamine', '1-aminopentane'], orca='pentylamine', g09='PentylAmine', nwchem='pentamin', mopac='pentylamine'),
            Solvent(name='perfluorobenzene', smiles='FC1=C(F)C(F)=C(F)C(F)=C1F', aliases=['perfluorobenzene', 'pfb', 'c6f6', 'hexafluorobenzene'], orca='perfluorobenzene', g09='PerFluoroBenzene', nwchem='pfb', mopac='perfluorobenzene'),
            Solvent(name='propanal', smiles='CCC=O', aliases=['propanal'], orca='propanal', g09='Propanal', nwchem='propanal', mopac='propanal'),
            Solvent(name='propanoic acid', smiles='CCC(O)=O', aliases=['propanoic acid', 'propionic acid'], orca='propanoic acid', g09='PropanoicAcid', nwchem='propacid', mopac='propanoic acid'),
            Solvent(name='propanenitrile', smiles='CCC#N', aliases=['propanenitrile', 'cyanoethane', 'ethyl cyanide', 'propanonitrile'], orca='propanonitrile', g09='PropanoNitrile', nwchem='propntrl', mopac='cyanoethane'),
            Solvent(name='propyl ethanoate', smiles='CC(OCCC)=O', aliases=['propyl ethanoate', 'propyl acetate'], orca='propyl ethanoate', g09='PropylEthanoate', nwchem='propacet', mopac='propyl acetate'),
            Solvent(name='propyl amine', smiles='NCCC', aliases=['propyl amine', 'propylamine', '1-aminopropane'], orca='propylamine', g09='PropylAmine', nwchem='propamin', mopac='propylamine'),
            Solvent(name='tetrachloroethene', smiles='Cl/C(Cl)=C(Cl)/Cl', aliases=['tetrachloroethene', 'perchloroethene', 'pce', 'c2cl4'], orca='tetrachloroethene', g09='TetraChloroEthene', nwchem='c2cl4', mopac='tetrachloroethene'),
            Solvent(name='tetrahydrothiophene-s,s-dioxide', smiles='O=S1(CCCC1)=O', aliases=['tetrahydrothiophene-s,s-dioxide', 'sulfolane'], orca='tetrahydrothiophene-s,s-dioxide', g09='TetraHydroThiophene-s,s-dioxide', nwchem='sulfolan', mopac='sulfolane'),
            Solvent(name='tetralin', smiles='C12=C(CCCC2)C=CC=C1', aliases=['tetralin', '1,2,3,4-tetrahydronaphthalene', 'tetrahydronaphthalene'], orca='tetralin', g09='Tetralin', nwchem='tetralin', mopac='tetralin'),
            Solvent(name='thiophene', smiles='C1=CC=CS1', aliases=['thiophene'], orca='thiophene', g09='Thiophene', nwchem='thiophen', mopac='thiophene'),
            Solvent(name='thiophenol', smiles='SC1=CC=CC=C1', aliases=['thiophenol', 'phsh', 'benzenethiol'], orca='thiophenol', g09='Thiophenol', nwchem='phsh', mopac='thiophenol'),
            Solvent(name='tributylphosphate', smiles='O=P(OCCCC)(OCCCC)OCCCC', aliases=['tributylphopshate', 'tbp', 'tributyl phopshate'], orca='tributylphopshate', g09='TriButylPhosphate', nwchem='tbp', mopac='tbp'),
            Solvent(name='trichloroethene', smiles='Cl/C(Cl)=C/Cl', aliases=['trichloroethene', 'tce'], orca='trichloroethene', g09='TriChloroEthene', nwchem='tce', mopac='tce'),
            Solvent(name='triethylamine', smiles='CCN(CC)CC', aliases=['triethylamine', 'et3n'], orca='triethylamine', g09='TriEthylAmine', nwchem='et3n', mopac='triethylamine'),
            Solvent(name='n-undecane', smiles='CCCCCCCCCCC', aliases=['n-undecane', 'undecane'], orca='n-undecane', g09='n-Undecane', nwchem='undecane', mopac='n-undecane'),
            Solvent(name='xylene mixture', smiles='CC1=CC=C(C)C=C1', aliases=['xylene mix', 'xylene (mix)', 'xylene mixture', 'xylene (mixture)', 'xylene'], orca='xyzlene (mixture)', g09='Xylene-mixture', nwchem='xylenemx', mopac='xylene mix'),
            Solvent(name='m-xylene', smiles='CC1=CC=CC(C)=C1', aliases=['m-xylene', 'meta-xylene', '1,3-xylene'], orca='m-xylene', g09='m-Xylene', nwchem='m-xylene', mopac='m-xylene'),
            Solvent(name='o-xylene', smiles='CC1=CC=CC=C1C', aliases=['o-xylene', 'ortho-xylene', '1,2-xylene'], orca='o-xylene', g09='o-Xylene', nwchem='o-xylene', mopac='o-xylene'),
            Solvent(name='p-xylene', smiles='CC1=CC=C(C)C=C1', aliases=['p-xylene', 'para-xylene', '1,4-xylene'], orca='p-xylene', g09='p-Xylene', nwchem='p-xylene', mopac='p-xylene'),
            Solvent(name='2-propanol', smiles='CC(O)C', aliases=['2-propanol', 'propan-2-ol', 'isopropanol', 'isopropyl alcohol'], orca='2-propanol', g09='2-Propanol', nwchem='propnol2', mopac='2-propanol'),
            Solvent(name='2-propen-1-ol', smiles='C=CCO', aliases=['2-propen-1-ol', 'allyl alcohol'], orca='2-propen-1-ol', g09='2-Propen-1-ol', nwchem='propenol', mopac='2-propen-1-ol'),
            Solvent(name='e-2-pentene', smiles='C/C=C/CC', aliases=['e-2-pentene', 'e-pent-2-ene'], orca='e-2-pentene', g09='e-2-Pentene', nwchem='e2penten', mopac='e-2-pentene'),
            Solvent(name='3-methylpyridine', smiles='CC1=CC=CN=C1', aliases=['3-methylpyridine', '3-picoline'], orca='3-methylpyridine', g09='3-MethylPyridine', nwchem='mepyrid3', mopac='3-methylpyridine'),
            Solvent(name='3-pentanone', smiles='CCC(CC)=O', aliases=['3-pentanone', 'pentan-3-one'], orca='3-pentanone', g09='3-Pentanone', nwchem='pentnon3', mopac='3-pentanone'),
            Solvent(name='4-heptanone', smiles='CCCC(CCC)=O', aliases=['4-heptanone', 'heptan-4-one'], orca='4-heptanone', g09='4-Heptanone', nwchem='heptnon4', mopac='4-heptanone'),
            Solvent(name='4-methyl-2-pentanone', smiles='CC(CC(C)C)=O', aliases=['4-methyl-2-pentanone', 'methyl isobutyl ketone'], orca='4-methyl-2-pentanone', g09='4-Methyl-2-Pentanone', nwchem='mibk', mopac='mibk'),
            Solvent(name='4=methylpyridine', smiles='CC1=CC=NC=C1', aliases=['4-methylpyridine', '4-picoline'], orca='4-methylpyridine', g09='4-MethylPyridine', nwchem='mepyrid4', mopac='4-methylpyridine'),
            Solvent(name='5-nonanone', smiles='CCCCC(CCCC)=O', aliases=['5-nonanone', 'nonan-5-one'], orca='5-nonanone', g09='5-Nonanone', nwchem='nonanone', mopac='5-nonanone'),
            Solvent(name='benzyl alcohol', smiles='OCC1=CC=CC=C1', aliases=['benzyl alcohol', 'phenylmethanol', 'bnoh'], orca='benzyl alcohol', g09='BenzylAlcohol', nwchem='benzalcl', mopac='benzyl alcohol'),
            Solvent(name='butanoic acid', smiles='CCCC(O)=O', aliases=['butanoic acid', 'butyric acid'], orca='butanoic acid', g09='ButanoicAcid', nwchem='butacid', mopac='butanoic acid'),
            Solvent(name='butanenitrile', smiles='CCCC#N', aliases=['butanenitrile', 'butyronitrile', 'butanonitrile'], orca='butanonitrile', g09='ButanoNitrile', nwchem='butantrl', mopac='butanenitrile'),
            Solvent(name='butyl ethanoate', smiles='CC(OCCCC)=O', aliases=['butyl ethanoate', 'butyl acetate'], orca='butyl ethanoate', g09='ButylEthanoate', nwchem='butile', mopac='butyl acetate'),
            Solvent(name='butylamine', smiles='NCCCC', aliases=['butylamine', 'butan-1-amine'], orca='butylamine', g09='ButylAmine', nwchem='nba', mopac='butylamine'),
            Solvent(name='n-butylbenzene', smiles='CCCCC1=CC=CC=C1', aliases=['n-butylbenzene', 'butylbenzene', 'phenylbutane'], orca='n-butylbenzene', g09='n-ButylBenzene', nwchem='nbutbenz', mopac='n-butylbenzene'),
            Solvent(name='sec-butylbenzene', smiles='CCC(C1=CC=CC=C1)C', aliases=['sec-butylbenzene', 's-butylbenzene'], orca='sec-butylbenzene', g09='sec-ButylBenzene', nwchem='sbutbenz', mopac='s-butylbenzene'),
            Solvent(name='tert-butylbenzene', smiles='CC(C1=CC=CC=C1)(C)C', aliases=['tert-butylbenzene', 't-butylbenzene'], orca='tert-butylbenzene', g09='tert-ButylBenzene', nwchem='tbutbenz', mopac='t-butylbenzene'),
            Solvent(name='o-chlorotoluene', smiles='CC1=CC=CC=C1Cl', aliases=['o-chlorotoluene', 'ortho-chlorotoluene', '2-chlorotoluene'], orca='o-chlorotoluene', g09='o-ChloroToluene', nwchem='ocltolue', mopac='o-chlorotoluene'),
            Solvent(name='m-cresol', smiles='CC1=CC(O)=CC=C1', aliases=['m-cresol', 'meta-cresol', '3-methylphenol'], orca='m-cresol', g09='m-Cresol', nwchem='m-cresol', mopac='m-cresol'),
            Solvent(name='o-cresol', smiles='CC1=CC=CC=C1O', aliases=['o-cresol', 'ortho-cresol', '2-methylphenol'], orca='o-cresol', g09='o-Cresol', nwchem='o-cresol', mopac='o-cresol'),
            Solvent(name='cyclohexanone', smiles='O=C1CCCCC1', aliases=['cyclohexanone'], orca='cyclohexanone', g09='CycloHexanone', nwchem='cychexon', mopac='cyclohexanone'),
            Solvent(name='isoquinoline', smiles='C12=C(C=NC=C2)C=CC=C1', aliases=['isoquinoline'], g09='IsoQuinoline', mopac='isoquinoline'),
            Solvent(name='quinoline', smiles='C12=CC=CC=C1N=CC=C2', aliases=['quinoline'], g09='Quinoline', mopac='quinoline'),
            Solvent(name='argon', smiles='[Ar]', aliases=['argon'], g09='Argon', mopac='argon'),
            Solvent(name='krypton', smiles='[Kr]', aliases=['krypton'], g09='Krypton', mopac='krypton'),
            Solvent(name='xenon', smiles='[Xe]', aliases=['xenon'], g09='Xenon', mopac='xenon')]
