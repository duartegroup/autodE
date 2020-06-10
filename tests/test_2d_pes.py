import autode.pes.pes_2d as pes_2d
import numpy as np
from autode.species.molecule import Reactant, Product
from autode.atoms import Atom
from autode.species.complex import ReactantComplex, ProductComplex
from autode.pes.pes import FormingBond, BreakingBond
from autode.reactions.reaction import Reaction
from autode.wrappers.XTB import xtb
from autode.config import Config
import os

Config.high_quality_plots = False
xtb.available = True

here = os.path.dirname(os.path.abspath(__file__))


def test_polyfit2d():
    x = [6, 2, 18, 12]
    y = [4, 5, 6, 7]
    z = [14, 12, 30, 26]
    coeff_mat = pes_2d.polyfit2d(x, y, z, order=1)
    assert len(coeff_mat) == 2
    assert coeff_mat.shape[0] == 2
    assert coeff_mat.shape[1] == 2
    assert -0.005 < coeff_mat[0, 0] < 0.005
    assert 1.995 < coeff_mat[0, 1] < 2.005
    assert 0.995 < coeff_mat[1, 0] < 1.005
    assert -0.005 < coeff_mat[1, 1] < 0.005


def test_get_ts_guess_2dscan():
    os.chdir(os.path.join(here, 'data'))
    Config.keep_input_files = False

    ch3cl_f = Reactant(name='CH3Cl_F-', charge=-1, mult=1,
                       atoms=[Atom('F', -4.14292, -0.24015,  0.07872),
                              Atom('Cl',  1.63463,  0.09787, -0.02490),
                              Atom('C', -0.14523, -0.00817,  0.00208),
                              Atom('H', -0.47498, -0.59594, -0.86199),
                              Atom('H', -0.45432, -0.49900,  0.93234),
                              Atom('H', -0.56010,  1.00533, -0.04754)])

    ch3f_cl = Product(name='CH3Cl_F-', charge=-1, mult=1,
                      atoms=[Atom('F',  1.63463,  0.09787, -0.02490),
                             Atom('Cl', -4.14292, -0.24015,  0.07872),
                             Atom('C', -0.14523, -0.00817,  0.00208),
                             Atom('H', -0.47498, -0.59594, -0.86199),
                             Atom('H', -0.45432, -0.49900,  0.93234),
                             Atom('H', -0.56010,  1.00533, -0.04754)])

    #         H                H
    #   F-   C--Cl     ->   F--C         Cl-
    #       H H               H H
    pes = pes_2d.PES2d(reactant=ReactantComplex(ch3cl_f),
                       product=ProductComplex(ch3f_cl),
                       r1s=np.linspace(4.0, 1.5, 9), r1_idxs=(0, 2),
                       r2s=np.linspace(1.78, 4.0, 8), r2_idxs=(1, 2))

    pes.calculate(name='SN2_PES', method=xtb, keywords=xtb.keywords.low_opt)

    assert pes.species[0, 1] is not None
    assert pes.species[0, 1].energy == -13.116895286939
    assert pes.species.shape == (9, 8)
    assert pes.rs.shape == (9, 8)
    assert type(pes.rs[0, 1]) == tuple
    assert pes.rs[1, 1] == (np.linspace(4.0, 1.5, 9)[1], np.linspace(1.78, 4.0, 8)[1])

    # Fitting the surface with a 2D polynomial up to order 3 in r1 and r2 i.e. r1^3r2^3
    pes.fit(polynomial_order=3)
    assert pes.coeff_mat is not None
    assert pes.coeff_mat.shape == (4, 4)        # Includes r1^0 etc.

    pes.print_plot(name='pes_plot')
    assert os.path.exists('pes_plot.png')

    # Products should be made on this surface
    assert pes.products_made()

    # Get the TS guess from this surface calling all the above functions
    reactant = ReactantComplex(ch3cl_f)
    fbond = FormingBond(atom_indexes=(0, 2), species=reactant)
    fbond.final_dist = 1.5

    bbond = BreakingBond(atom_indexes=(1, 2), species=reactant, reaction=Reaction(ch3cl_f, ch3f_cl))
    bbond.final_dist = 4.0

    ts_guess = pes_2d.get_ts_guess_2d(reactant=reactant, product=ProductComplex(ch3f_cl),
                                      bond1=fbond, bond2=bbond, polynomial_order=3,
                                      name='SN2_PES',
                                      method=xtb,
                                      keywords=xtb.keywords.low_opt,
                                      dr=0.3)
    assert ts_guess is not None
    assert ts_guess.n_atoms == 6
    assert ts_guess.energy is None
    assert 2.13 < ts_guess.get_distance(0, 2) < 3.14
    assert 1.9 < ts_guess.get_distance(1, 2) < 2.0

    for filename in os.listdir(os.getcwd()):
        if filename.endswith(('.inp', '.png')) or 'animation' in filename or 'xcontrol' in filename:
            os.remove(filename)

    os.chdir(here)
    Config.keep_input_files = True
