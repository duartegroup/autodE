import autode as ade

orca = ade.methods.ORCA()

if not orca.available:
    exit('This example requires an ORCA install')

# By default, all ab-initio calculations with high-level electronic structure
# codes are performed at PBE0-D3BJ/def2-S/TZVP. For example,
h2 = ade.Molecule(smiles='[H][H]')

print(f'Using {ade.Config.n_cores} cores for the calculations')
h2.optimise(method=orca)            # PBE0-D3BJ/def2-SVP
h2.single_point(method=orca)        # PBE0-D3BJ/def2-TZVP

# This can be changed by setting keywords. e.g. for an ORCA B3lYP single point
h2.single_point(method=orca,
                keywords=['SP', 'B3LYP', 'def2-TZVP'])

# The global configuration for all calculations using a method can be set
# with ade.Config. To set the optimisation keywords suitable for a PBE/def2-SVP
# calculation in Gaussian09
ade.Config.G09.keywords.opt = ['PBEPBE', 'Def2SVP', 'integral=ultrafinegrid']

g09 = ade.methods.G09()
if not g09.available:
    exit('This part requires a Gassian09 install')

h2.optimise(method=g09)  # Uses PBE/def2-SVP

n2 = ade.Molecule(smiles='N#N')
n2.optimise(method=g09)  # also uses PBE/def2-SVP
