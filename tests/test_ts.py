from autode.atoms import Atom
from autode.transition_states.templates import get_ts_templates
from autode.transition_states.ts_guess import TSguess
from autode.bond_rearrangement import BondRearrangement
from autode.reactions.reaction import Reaction
from autode.transition_states.transition_state import TransitionState
from autode.species.molecule import Reactant, Product
from autode.species.complex import ReactantComplex, ProductComplex
from autode.config import Config
from autode.calculation import Calculation
from autode.wrappers.ORCA import ORCA
from autode.transition_states.base import imag_mode_links_reactant_products
from autode.transition_states.base import imag_mode_has_correct_displacement
from autode.transition_states.base import imag_mode_generates_other_bonds
from autode.species.species import Species
from autode.transition_states.base import get_displaced_atoms_along_mode
from autode.wrappers.G09 import G09
import os
here = os.path.dirname(os.path.abspath(__file__))
method = ORCA()
method.available = True

# Force ORCA to appear available
Config.hcode = 'orca'
Config.ORCA.path = here

ch3cl = Reactant(charge=0, mult=1, atoms=[Atom('Cl', 1.63664, 0.02010, -0.05829),
                                          Atom('C', -0.14524, -0.00136, 0.00498),
                                          Atom('H', -0.52169, -0.54637, -0.86809),
                                          Atom('H', -0.45804, -0.50420, 0.92747),
                                          Atom('H', -0.51166, 1.03181, -0.00597)])
f = Reactant(charge=-1, mult=1, atoms=[Atom('F', 4.0, 0.0, 0.0)])
reac_complex = ReactantComplex(f, ch3cl)

ch3f = Product(charge=0, mult=1, atoms=[Atom('C', -0.05250, 0.00047, -0.00636),
                                        Atom('F', 1.31229, -0.01702, 0.16350),
                                        Atom('H', -0.54993, -0.04452, 0.97526),
                                        Atom('H', -0.34815, 0.92748, -0.52199),
                                        Atom('H', -0.36172, -0.86651, -0.61030)])
cl = Product(charge=-1, mult=1, atoms=[Atom('Cl', 4.0, 0.0, 0.0)])
product_complex = ProductComplex(ch3f, cl)

tsguess = TSguess(reactant=reac_complex, product=product_complex,
                  atoms=[Atom('F', -2.66092, -0.01426, 0.09700),
                         Atom('Cl', 1.46795, 0.05788, -0.06166),
                         Atom('C', -0.66317, -0.01826, 0.02488),
                         Atom('H', -0.78315, -0.58679, -0.88975),
                         Atom('H', -0.70611, -0.54149, 0.97313),
                         Atom('H', -0.80305, 1.05409, 0.00503)])

tsguess.bond_rearrangement = BondRearrangement(breaking_bonds=[(2, 1)],
                                               forming_bonds=[(0, 2)])

ts = TransitionState(ts_guess=tsguess)


def test_ts_guess_class():
    os.chdir(os.path.join(here, 'data'))

    # Force ORCA to appear available
    Config.hcode = 'orca'
    Config.ORCA.path = here

    assert tsguess.reactant.n_atoms == 6
    assert tsguess.product.n_atoms == 6

    # C -- Cl distance should be long
    assert tsguess.product.get_distance(0, 5) > 3.0

    assert tsguess.calc is None
    assert hasattr(tsguess, 'bond_rearrangement')
    assert tsguess.bond_rearrangement is not None

    # TS guess should at least initially only have the bonds in the reactant
    assert tsguess.graph.number_of_edges() == 4

    assert tsguess.could_have_correct_imag_mode(method=method)
    assert tsguess.has_correct_imag_mode()

    os.chdir(here)


def test_links_reacs_prods():
    os.chdir(os.path.join(here, 'data'))

    tsguess.calc = Calculation(name=tsguess.name + '_hess', molecule=tsguess, method=method,
                               keywords=method.keywords.hess, n_cores=Config.n_cores)
    # Should find the completed calculation output
    tsguess.calc.run()

    # Spoof an xtb install as reactant/product complex optimisation
    Config.lcode = 'xtb'
    Config.XTB.path = here

    Config.num_complex_sphere_points = 4
    Config.num_complex_random_rotations = 1

    assert imag_mode_links_reactant_products(calc=tsguess.calc,
                                             reactant=reac_complex,
                                             product=product_complex,
                                             method=method)

    for i in range(4):
        os.remove(f'complex_conf{i}.xyz')
        os.remove(f'complex_conf{i}_opt_xtb.xyz')

    os.remove('ts_guess_hess_orca.inp')
    os.remove('ts_guess_hess_orca_forwards_orca.inp')
    os.remove('ts_guess_hess_orca_backwards_orca.inp')

    os.chdir(here)


def test_correct_imag_mode():
    os.chdir(os.path.join(here, 'data'))

    bond_rearrangement = BondRearrangement(breaking_bonds=[(4, 1), (4, 18)],
                                           forming_bonds=[(1, 18)])
    g09 = G09()
    g09.available = True

    calc = Calculation(name='tmp', molecule=ReactantComplex(Reactant(smiles='CC(C)(C)C1C=CC=C1')),
                       method=g09, keywords=Config.G09.keywords.opt_ts)
    calc.output.filename = 'correct_ts_mode_g09.log'
    calc.output.set_lines()

    f_displaced_atoms = get_displaced_atoms_along_mode(calc, mode_number=6, disp_magnitude=1.0)
    f_species = Species(name='f_displaced', atoms=f_displaced_atoms, charge=0, mult=1)  # Charge & mult are placeholders

    b_displaced_atoms = get_displaced_atoms_along_mode(calc, mode_number=6, disp_magnitude=-1.0)
    b_species = Species(name='b_displaced', atoms=b_displaced_atoms, charge=0, mult=1)

    # With the correct mode no other bonds are made
    assert not imag_mode_generates_other_bonds(ts=calc.molecule, f_species=f_species, b_species=b_species,
                                               bond_rearrangement=bond_rearrangement)

    calc.output.filename = 'incorrect_ts_mode_g09.log'
    calc.output.set_lines()

    assert not imag_mode_has_correct_displacement(calc, bond_rearrangement)

    os.chdir(here)


def test_isomorphic_reactant_product():

    os.chdir(os.path.join(here, 'data', 'locate_ts'))

    r_water = Reactant(name='h2o', smiles='O')
    r_methane = Reactant(name='methane', smiles='C')

    p_water = Product(name='h2o', smiles='O')
    p_methane = Product(name='methane', smiles='C')

    # Reaction where the reactant and product complexes are isomorphic should return no TS
    reaction = Reaction(r_water, r_methane, p_water, p_methane)
    reaction.locate_transition_state()

    os.chdir(here)

    assert reaction.ts is None


def test_find_tss():

    os.chdir(os.path.join(here, 'data', 'locate_ts'))
    Config.num_conformers = 1

    # Spoof ORCA and XTB installs
    Config.ORCA.path = here
    Config.XTB.path = '/home/tom/.local/bin/xtb'# here
    Config.ORCA.implicit_solvation_type = 'cpcm'
    Config.make_ts_template = False
    Config.num_complex_sphere_points = 2
    Config.num_complex_random_rotations = 1

    # Simple rearrangement reaction
    r = Reactant(smiles='CC[C]([H])[H]', name='r')
    p1 = Product(smiles='C[C]([H])C', name='p1')

    reaction = Reaction(r, p1, solvent_name='water')
    # Will work in data/locate_ts/transition_states
    reaction.locate_transition_state()

    assert reaction.ts is not None
    os.chdir(os.path.join(here, 'data', 'locate_ts', 'transition_states'))

    for filename in os.listdir(os.getcwd()):
        if filename.endswith(('.inp', '.png')):
            os.remove(filename)

    assert reaction.ts.is_true_ts()

    reaction.ts.save_ts_template(folder_path=os.getcwd())
    assert os.path.exists('template0.obj')

    # There should now be a saved template
    templates = get_ts_templates(folder_path=os.getcwd())
    assert len(templates) == 1

    template = templates[0]
    assert template.solvent.name == 'water'
    assert template.mult == 2
    assert template.charge == 0

    # Truncated graph has 7 atoms in
    assert template.graph.number_of_nodes() == 7

    os.remove('template0.obj')
    os.chdir(here)


def test_ts_templates():

    templates = get_ts_templates(folder_path='/a/path/that/doesnt/exist')
    assert len(templates) == 0
