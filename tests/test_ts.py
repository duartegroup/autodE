# from autode.transition_states import ts_guess
# from autode.atoms import Atom
# from autode.transition_states.templates import get_ts_templates
# from autode.transition_states.ts_guess import TSguess
# from autode.bond_rearrangement import BondRearrangement
# from autode.transition_states.transition_state import TransitionState
# from autode.reactions import Substitution
# from autode import molecule
# from autode.molecule import Reactant, Product
# from autode.complex import ReactantComplex, ProductComplex
# from autode.config import Config
# from autode.wrappers.ORCA import ORCA
# from autode.solvent.solvents import Solvent
# import os
# here = os.path.dirname(os.path.abspath(__file__))

# Config.make_ts_template = False

# ch3cl = Reactant(charge=0, mult=1, atoms=[Atom('Cl', 1.63664, 0.02010, -0.05829),
#                                           Atom('C', -0.14524, -0.00136, 0.00498),
#                                           Atom('H', -0.52169, -0.54637, -0.86809),
#                                           Atom('H', -0.45804, -0.50420, 0.92747),
#                                           Atom('H', -0.51166, 1.03181, -0.00597)])

# f = Reactant(charge=-1, mult=1, atoms=[Atom('F', 4.0, 0.0, 0.0)])

# ch3f = Product(charge=0, mult=1, atoms=[Atom('C', -0.05250, 0.00047, -0.00636),
#                                         Atom('F', 1.31229, -0.01702, 0.16350),
#                                         Atom('H', -0.54993, -0.04452, 0.97526),
#                                         Atom('H', -0.34815, 0.92748, -0.52199),
#                                         Atom('H', -0.36172, -0.86651, -0.61030)])

# cl = Reactant(charge=-1, mult=1, atoms=[Atom('Cl', 4.0, 0.0, 0.0)])

# f_ch3cl_tsguess = TSguess(reactant=ReactantComplex(f, ch3cl),
#                           product=ProductComplex(ch3f, cl),
#                           atoms=[Atom('F', -2.66092, -0.01426, 0.09700),
#                                  Atom('Cl', 1.46795, 0.05788, -0.06166),
#                                  Atom('C', -0.66317, -0.01826, 0.02488),
#                                  Atom('H', -0.78315, -0.58679, -0.88975),
#                                  Atom('H', -0.70611, -0.54149, 0.97313),
#                                  Atom('H', -0.80305, 1.05409, 0.00503)])

# f_ch3cl_tsguess.bond_rearrangement = BondRearrangement(breaking_bonds=[(2, 1)],
#                                                        forming_bonds=[(0, 2)])

# f_ch3cl_ts = TransitionState(ts_guess=f_ch3cl_tsguess)


# def test_ts_guess_class():

#     pass
