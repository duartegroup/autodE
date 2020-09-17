import os
from . import testutils
import pytest
from autode.exceptions import TemplateLoadingFailed
from autode.config import Config
from autode.bond_rearrangement import BondRearrangement
from autode.species.complex import ReactantComplex, ProductComplex
from autode.species.molecule import Reactant, Product
from autode.atoms import Atom
from autode.transition_states.templates import get_ts_templates
from autode.transition_states.templates import TStemplate
from autode.mol_graphs import get_truncated_active_mol_graph
from autode.solvent.solvents import get_solvent
from autode.transition_states.ts_guess import get_template_ts_guess
from autode.wrappers.XTB import XTB
here = os.path.dirname(os.path.abspath(__file__))


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


@testutils.work_in_zipped_dir(os.path.join(here, 'data', 'ts_guess.zip'))
def test_ts_template_save():

    ts_graph = reac_complex.graph.copy()

    # Add the F-C bond as active
    ts_graph.add_edge(0, 2, active=True)

    # Remove then re-add the C-Cl bond as active
    ts_graph.remove_edge(1, 2)
    ts_graph.add_edge(1, 2, active=True)

    truncated_graph = get_truncated_active_mol_graph(ts_graph)

    template = TStemplate(truncated_graph, species=reac_complex)
    template.save(folder_path=os.getcwd())

    assert os.path.exists('template0.txt')

    # With no distances the template shouldn't be valid
    with pytest.raises(TemplateLoadingFailed):
        _ = TStemplate(filename='template0.txt')

    os.remove('template0.txt')

    truncated_graph.edges[(0, 2)]['distance'] = 1.9
    truncated_graph.edges[(1, 2)]['distance'] = 2.0

    template.graph = truncated_graph
    template.save(folder_path=os.getcwd())
    loaded_template = TStemplate(filename='template_sn2.txt')

    assert loaded_template.solvent is None
    assert loaded_template.charge == -1
    assert loaded_template.mult == 1

    assert loaded_template.graph is not None
    assert loaded_template.graph.nodes == truncated_graph.nodes
    assert loaded_template.graph.edges == truncated_graph.edges


@testutils.work_in_zipped_dir(os.path.join(here, 'data', 'ts_guess.zip'))
def test_ts_template():

    # Spoof XTB install
    Config.XTB.path = here
    Config.ts_template_folder_path = os.path.join(here, 'data', 'ts_guess')

    bond_rearr = BondRearrangement(breaking_bonds=[(2, 1)],
                                   forming_bonds=[(0, 2)])

    reac_shift = reac_complex.copy()

    reac_shift.set_atoms(atoms=[Atom('F', -3.0587, -0.8998, -0.2180),
                                Atom('Cl', 0.3842, 0.86572,-1.65507),
                                Atom('C', -1.3741, -0.0391, -0.9719),
                                Atom('H', -1.9151, -0.0163, -1.9121),
                                Atom('H', -1.6295,  0.6929, -0.2173),
                                Atom('H', -0.9389, -0.9786, -0.6534)])
    reac_shift.print_xyz_file()

    templates = get_ts_templates()
    assert len(templates) == 1
    assert templates[0].graph.number_of_nodes() == 6

    tsg_template = get_template_ts_guess(reac_shift, product_complex,
                                         name='template',
                                         bond_rearr=bond_rearr,
                                         method=XTB(),
                                         dist_thresh=4.0)

    # Reset the folder path to the default
    Config.ts_template_folder_path = None

    assert tsg_template is not None


def test_ts_templates_find():

    templates = get_ts_templates(folder_path='/a/path/that/doesnt/exist')
    assert len(templates) == 0
