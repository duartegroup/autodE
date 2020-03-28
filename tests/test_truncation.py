from autode.transition_states.truncation import strip_non_core_atoms
from autode.mol_graphs import is_isomorphic
from autode.complex import ReactantComplex
from autode.molecule import Reactant
from autode.atoms import Atom

methane = Reactant(name='methane', charge=0, mult=1,
                   atoms=[Atom('C', 0.93919, -0.81963, 0.00000),
                          Atom('H', 2.04859, -0.81963, -0.00000),
                          Atom('H', 0.56939, -0.25105, 0.87791),
                          Atom('H', 0.56938, -1.86422, 0.05345),
                          Atom('H', 0.56938, -0.34363, -0.93136)])

ethene = Reactant(name='ethene', charge=0, mult=1,
                  atoms=[Atom('C',  0.84102, -0.74223,  0.00000),
                         Atom('C', -0.20368,  0.08149,  0.00000),
                         Atom('H',  1.63961, -0.61350, -0.72376),
                         Atom('H',  0.90214, -1.54881,  0.72376),
                         Atom('H', -0.26479,  0.88807, -0.72376),
                         Atom('H', -1.00226, -0.04723,  0.72376)])

propene = Reactant(name='propene', charge=0, mult=1,
                   atoms=[Atom('C',  1.06269, -0.71502,  0.09680),
                          Atom('C',  0.01380,  0.10714,  0.00458),
                          Atom('H',  0.14446,  1.16840, -0.18383),
                          Atom('H', -0.99217, -0.28355,  0.11871),
                          Atom('C',  2.47243, -0.22658, -0.05300),
                          Atom('H',  0.89408, -1.77083,  0.28604),
                          Atom('H',  2.51402,  0.86756, -0.24289),
                          Atom('H',  2.95379, -0.75333, -0.90290),
                          Atom('H',  3.03695, -0.44766,  0.87649)])

but1ene = Reactant(name='but-1-ene', charge=0, mult=1,
                   atoms=[Atom('C',  1.32424, -0.75672,  0.09135),
                          Atom('C',  0.19057, -0.05301,  0.04534),
                          Atom('H',  0.20022,  1.00569, -0.19632),
                          Atom('H', -0.75861, -0.53718,  0.25084),
                          Atom('C',  2.64941, -0.10351, -0.19055),
                          Atom('H',  1.27555, -1.81452,  0.33672),
                          Atom('C',  3.79608, -1.10307, -0.07851),
                          Atom('H',  2.81589,  0.72011,  0.53717),
                          Atom('H',  2.63913,  0.32062, -1.21799),
                          Atom('H',  3.83918, -1.52715,  0.94757),
                          Atom('H',  4.75802, -0.59134, -0.29261),
                          Atom('H',  3.66134, -1.92824, -0.81028)])

benzene = Reactant(name='benzene', charge=0, mult=1, smiles='c1ccccc1')


def test_core_strip():

    stripped = strip_non_core_atoms(methane, active_atoms=[0])

    # Should not strip any atoms if the carbon is designated as active
    assert stripped.n_atoms == 5

    stripped = strip_non_core_atoms(ethene, active_atoms=[0])
    assert stripped.n_atoms == 6

    # Propene should strip to ethene if the terminal C=C is the active atom
    stripped = strip_non_core_atoms(propene, active_atoms=[1])
    assert stripped.n_atoms == 6
    assert is_isomorphic(stripped.graph, ethene.graph)

    # But-1-ene should strip to ethene if the terminal C=C is the active atom
    stripped = strip_non_core_atoms(but1ene, active_atoms=[1])
    assert stripped.n_atoms == 6
    assert is_isomorphic(stripped.graph, ethene.graph)

    # Benzene shouldn't be truncated at all
    stripped = strip_non_core_atoms(benzene, active_atoms=[1])
    assert stripped.n_atoms == 12


def test_reactant_complex_truncation():

    dimer = ReactantComplex(methane, methane)
