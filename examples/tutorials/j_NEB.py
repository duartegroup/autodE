import multiprocessing
import autode as ade

orca = ade.methods.ORCA()

# Set the keywords so autodE can extract gradients at PBE/def2-SV(P)
orca.keywords.grad = ["PBE", "def2-SV(P)", "EnGrad"]

if multiprocessing.cpu_count() < 10 or not orca.is_available:
    exit("This example requires an ORCA install and 10 processing cores")

# Nudged elastic band (NEB) calculations are available using all methods
# that support gradient evaluations (all of them!). For example, to
# set up a set of images and relax to the ~minimum energy path for a
# Diels Alder reaction between benzoquinone and cyclopentadiene
neb = ade.neb.NEB(
    initial_species=ade.Molecule("_data/DielsAlder/reactant.xyz"),
    final_species=ade.Molecule("_data/DielsAlder/product.xyz"),
    num=5,
)

neb.calculate(method=orca, n_cores=10)
# will have generated a plot of the relaxation, along with a .xyz
# trajectory of the initial and final NEB path

# To use a climbing image NEB simply replace ade.neb.NEB with ade.neb.CINEB
