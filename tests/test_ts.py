from autode.atoms import Atom
from autode.transition_states.templates import get_ts_templates
from autode.transition_states.ts_guess import TSguess
from autode.transition_states.base import f_b_isomorphic_to_r_p
from autode.bond_rearrangement import BondRearrangement
from autode.reactions.reaction import Reaction
from autode.transition_states.transition_state import TransitionState
from autode.input_output import xyz_file_to_atoms
from autode.species.molecule import Reactant, Product
from autode.species.complex import ReactantComplex, ProductComplex
from autode.config import Config
from autode.calculation import Calculation
from autode.wrappers.ORCA import ORCA
from autode.wrappers.implicit_solvent_types import cpcm
from autode.transition_states.base import imag_mode_generates_other_bonds
from autode.transition_states.base import displaced_species_along_mode
from autode.wrappers.G09 import G09
from . import testutils
import pytest
import os
import shutil


here = os.path.dirname(os.path.abspath(__file__))
method = ORCA()

# Force ORCA to appear available
Config.hcode = 'orca'
Config.ORCA.path = here

ch3cl = Reactant(charge=0, mult=1,
                 atoms=[Atom('Cl', 1.63664, 0.02010, -0.05829),
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


@testutils.work_in_zipped_dir(os.path.join(here, 'data', 'ts.zip'))
def test_ts_guess_class():

    # Force ORCA to appear available
    Config.hcode = 'orca'
    Config.ORCA.path = here

    assert tsguess.reactant.n_atoms == 6
    assert tsguess.product.n_atoms == 6

    # C -- Cl distance should be long
    assert tsguess.product.distance(0, 5) > 3.0

    assert hasattr(tsguess, 'bond_rearrangement')
    assert tsguess.bond_rearrangement is not None

    # TS guess should at least initially only have the bonds in the reactant
    assert tsguess.graph.number_of_edges() == 4

    assert tsguess.could_have_correct_imag_mode
    assert tsguess.has_correct_imag_mode

    # Cannot check the imaginary mode without a bond rearrangment
    with pytest.raises(ValueError):
        _ = TSguess(atoms=tsguess.atoms).could_have_correct_imag_mode


@testutils.work_in_zipped_dir(os.path.join(here, 'data', 'ts.zip'))
@testutils.requires_with_working_xtb_install
def test_links_reacs_prods():

    Config.lcode = 'xtb'

    Config.num_complex_sphere_points = 4
    Config.num_complex_random_rotations = 1

    method.path = here
    assert method.available

    tsguess._run_hess_calculation(method=method)
    assert tsguess.imag_mode_links_reactant_products()


@testutils.work_in_zipped_dir(os.path.join(here, 'data', 'mode_checking.zip'))
def test_correct_imag_mode():

    bond_rearr = BondRearrangement(breaking_bonds=[(4, 1), (4, 18)],
                                   forming_bonds=[(1, 18)])

    g09 = G09()
    reac = ReactantComplex(Reactant(smiles='CC(C)(C)C1C=CC=C1'))

    calc = Calculation(name='tmp',
                       molecule=reac,
                       method=g09,
                       keywords=Config.G09.keywords.opt_ts)

    calc.output.filename = 'correct_ts_mode_g09.log'

    ts_guess = TSguess(atoms=calc.get_final_atoms(), bond_rearr=bond_rearr)
    ts_guess.hessian = calc.get_hessian()
    assert ts_guess.has_imaginary_frequencies

    f_species = displaced_species_along_mode(ts_guess,
                                             mode_number=6, disp_factor=1.0)

    b_species = displaced_species_along_mode(ts_guess,
                                             mode_number=6, disp_factor=-1.0)

    # With the correct mode no other bonds are made
    assert not imag_mode_generates_other_bonds(ts=ts_guess,
                                               f_species=f_species,
                                               b_species=b_species)


@testutils.work_in_zipped_dir(os.path.join(here, 'data', 'mode_checking.zip'))
def test_incorrect_imag_mode():

    g09 = G09()
    reac = ReactantComplex(Reactant(smiles='CC(C)(C)C1C=CC=C1'))
    prod = ProductComplex(Product(smiles='CC1C=CC=C1'),
                          Product(smiles='C[C]C'))

    calc = Calculation(name='tmp',
                       molecule=reac,  # <- not really right
                       method=g09,
                       keywords=Config.G09.keywords.opt_ts)

    calc.output.filename = 'incorrect_ts_mode_g09.log'

    bond_rearr = BondRearrangement(breaking_bonds=[(4, 1), (4, 18)],
                                   forming_bonds=[(1, 18)])
    ts_guess = TSguess(atoms=calc.get_final_atoms(),
                       bond_rearr=bond_rearr,
                       reactant=reac,
                       product=prod)

    ts_guess.hessian = calc.get_hessian()

    # Should not need to run a QRC, as the mode generates a bond that is not
    # acitve
    assert not ts_guess.has_correct_imag_mode


@testutils.work_in_zipped_dir(os.path.join(here, 'data', 'locate_ts.zip'))
def test_isomorphic_reactant_product():

    r_water = Reactant(name='h2o', smiles='O')
    r_methane = Reactant(name='methane', smiles='C')

    p_water = Product(name='h2o', smiles='O')
    p_methane = Product(name='methane', smiles='C')

    # Reaction where the reactant and product complexes are isomorphic
    # should return no TS
    reaction = Reaction(r_water, r_methane, p_water, p_methane)

    with pytest.raises(ValueError):
        reaction.locate_transition_state()


@testutils.work_in_zipped_dir(os.path.join(here, 'data', 'locate_ts.zip'))
@testutils.requires_with_working_xtb_install
def test_find_tss():

    Config.num_conformers = 1

    # Spoof ORCA install
    Config.ORCA.path = here

    if os.path.exists('/dev/shm'):
        Config.ll_tmp_dir = '/dev/shm'

    Config.XTB.path = shutil.which('xtb')

    Config.ORCA.implicit_solvation_type = cpcm
    Config.make_ts_template = False
    Config.num_complex_sphere_points = 2
    Config.num_complex_random_rotations = 1

    # SN2 example
    flouride = Reactant(name='F-', smiles='[F-]')
    methyl_chloride = Reactant(name='CH3Cl', smiles='ClC')
    chloride = Product(name='Cl-', smiles='[Cl-]')
    methyl_flouride = Product(name='CH3F', smiles='CF')

    reaction = Reaction(flouride, methyl_chloride, chloride, methyl_flouride,
                        name='sn2', solvent_name='water')

    # Will work in data/locate_ts/transition_states
    reaction.locate_transition_state()

    assert reaction.ts is not None
    os.chdir('transition_states')
    assert reaction.ts.is_true_ts
    os.chdir('..')

    reaction.ts.save_ts_template(folder_path=os.getcwd())
    assert os.path.exists('template0.txt')

    # There should now be a saved template
    templates = get_ts_templates(folder_path=os.getcwd())
    assert len(templates) == 1

    template = templates[0]
    assert template.solvent.name == 'water'
    assert template.mult == 1
    assert template.charge == -1

    assert template.graph.number_of_nodes() == 6

    # Reset the configuration
    Config.ll_tmp_dir = None


@testutils.work_in_zipped_dir(os.path.join(here, 'data', 'ts.zip'))
def test_optts_no_reactants_products():

    da_ts_guess = TSguess(atoms=xyz_file_to_atoms('da_TS_guess.xyz'),
                          bond_rearr=BondRearrangement(forming_bonds=[(0, 5), (3, 4)]))
    da_ts = TransitionState(da_ts_guess)
    da_ts.optimise()

    assert da_ts.has_imaginary_frequencies
    assert len(da_ts.imaginary_frequencies) == 1
    imag_freq = da_ts.imaginary_frequencies[0]

    assert -500 < imag_freq < -300      # cm-1

    assert da_ts.could_have_correct_imag_mode
    assert da_ts.has_correct_imag_mode


def test_no_graph():

    ts_no_graph = TSguess(atoms=None)
    assert ts_no_graph.graph is None


def test_fb_rp_isomorphic():

    reac = ReactantComplex(f, ch3cl)
    prod = ProductComplex(ch3f, cl)

    assert f_b_isomorphic_to_r_p(forwards=prod,
                                 backwards=reac,
                                 reactant=reac,
                                 product=prod)

    assert f_b_isomorphic_to_r_p(forwards=reac,
                                 backwards=prod,
                                 reactant=reac,
                                 product=prod)

    assert not f_b_isomorphic_to_r_p(forwards=reac,
                                     backwards=reac,
                                     reactant=reac,
                                     product=prod)

    # Check for e.g. failed optimisation of the forwards displaced complex
    mol_no_atoms = Product()
    assert not f_b_isomorphic_to_r_p(forwards=mol_no_atoms,
                                     backwards=reac,
                                     reactant=reac,
                                     product=prod)


@testutils.work_in_zipped_dir(os.path.join(here, 'data', 'ts_truncation.zip'))
@testutils.requires_with_working_xtb_install
def test_truncated_ts():

    # Spoof ORCA install
    Config.ORCA.path = here
    Config.make_ts_template = True
    Config.ts_template_folder_path = os.getcwd()

    if os.path.exists('/dev/shm'):
        Config.ll_tmp_dir = '/dev/shm'

    Config.XTB.path = shutil.which('xtb')

    reaction = Reaction(smiles='CCCCCCF.[Cl-]>>CCCCCCCl.[F-]',
                        solvent_name='water')
    reaction.locate_transition_state()

    # locate TS should assign a TS as linking reactants and products, so
    # checking that the TS exists is sufficient
    assert reaction.ts is not None
