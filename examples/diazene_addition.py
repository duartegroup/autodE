from autode import *
from autode.transition_states.locate_tss import *
import os
from autode.methods import get_lmethod

Config.n_cores = 4

dimethylbut_two_ene = Reactant(name='2,3-dimethylbut-2-ene', smiles='CC(C)=C(C)C', solvent='water')
diazene = Reactant(name='diazene', smiles='N=N', solvent='water')
dimethylbutane = Product(name='2,3-dimethylbutane', smiles='CC(C(C)C)C', solvent='water')
nitrogen = Product(name='N2', smiles='N#N', solvent='water')

reaction = Reaction(dimethylbut_two_ene, diazene, dimethylbutane, nitrogen, name='diazene_addition')

here = os.getcwd()
directory_path = os.path.join(here, reaction.name)
if not os.path.isdir(directory_path):
    os.mkdir(directory_path)
os.chdir(directory_path)

reaction.find_lowest_energy_conformers()
reaction.optimise_reacs_prods()
reactant, product = get_reactant_and_product_complexes(reaction)
bond_rearrangs = get_bond_rearrangs(reactant, product)
for bond_rearrangement in bond_rearrangs:
    if reaction.type in [Substitution, Elimination]:
        set_complex_xyzs_translated_rotated(reactant, reaction.reacs, bond_rearrangement)

reactant.single_point(method=get_lmethod())
            