import os
import shutil
import numpy as np
from .. import testutils
import pytest
from autode.exceptions import TemplateLoadingFailed
from autode.config import Config
from autode.species.molecule import Molecule
from autode.bond_rearrangement import BondRearrangement
from autode.species.complex import ReactantComplex, ProductComplex
from autode.species.molecule import Reactant, Product
from autode.atoms import Atom
from autode.utils import work_in_tmp_dir
from autode.transition_states.templates import get_ts_templates
from autode.transition_states.templates import get_value_from_file
from autode.transition_states.templates import get_values_dict_from_file
from autode.transition_states.templates import TStemplate
from autode.transition_states.transition_state import TransitionState
from autode.transition_states.ts_guess import TSguess
from autode.mol_graphs import get_truncated_active_mol_graph
from autode.transition_states.ts_guess import get_template_ts_guess
from autode.input_output import xyz_file_to_atoms
from autode.wrappers.XTB import XTB

here = os.path.dirname(os.path.abspath(__file__))


ch3cl = Reactant(
    charge=0,
    mult=1,
    atoms=[
        Atom("Cl", 1.63664, 0.02010, -0.05829),
        Atom("C", -0.14524, -0.00136, 0.00498),
        Atom("H", -0.52169, -0.54637, -0.86809),
        Atom("H", -0.45804, -0.50420, 0.92747),
        Atom("H", -0.51166, 1.03181, -0.00597),
    ],
)
f = Reactant(charge=-1, mult=1, atoms=[Atom("F", 4.0, 0.0, 0.0)])
reac_complex = ReactantComplex(f, ch3cl)

ch3f = Product(
    charge=0,
    mult=1,
    atoms=[
        Atom("C", -0.05250, 0.00047, -0.00636),
        Atom("F", 1.31229, -0.01702, 0.16350),
        Atom("H", -0.54993, -0.04452, 0.97526),
        Atom("H", -0.34815, 0.92748, -0.52199),
        Atom("H", -0.36172, -0.86651, -0.61030),
    ],
)
cl = Product(charge=-1, mult=1, atoms=[Atom("Cl", 4.0, 0.0, 0.0)])
product_complex = ProductComplex(ch3f, cl)


@testutils.work_in_zipped_dir(os.path.join(here, "data", "ts_guess.zip"))
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

    assert os.path.exists("template0.txt")

    # With no distances the template shouldn't be valid
    with pytest.raises(TemplateLoadingFailed):
        _ = TStemplate(filename="template0.txt")

    os.remove("template0.txt")

    truncated_graph.edges[(0, 2)]["distance"] = 1.9
    truncated_graph.edges[(1, 2)]["distance"] = 2.0

    template.graph = truncated_graph
    template.save(folder_path=os.getcwd())
    loaded_template = TStemplate(filename="template_sn2.txt")

    assert loaded_template.solvent is None
    assert loaded_template.charge == -1
    assert loaded_template.mult == 1

    assert loaded_template.graph is not None
    assert loaded_template.graph.nodes == truncated_graph.nodes
    assert loaded_template.graph.edges == truncated_graph.edges


@testutils.work_in_zipped_dir(os.path.join(here, "data", "ts_guess.zip"))
def test_ts_template():

    # Spoof XTB install, if not installed
    if shutil.which("xtb") is None:
        Config.XTB.path = here

    Config.ts_template_folder_path = os.path.join(here, "data", "ts_guess")

    bond_rearr = BondRearrangement(
        breaking_bonds=[(2, 1)], forming_bonds=[(0, 2)]
    )

    reac_shift = reac_complex.copy()

    reac_shift.atoms = [
        Atom("F", -3.0587, -0.8998, -0.2180),
        Atom("Cl", 0.3842, 0.86572, -1.65507),
        Atom("C", -1.3741, -0.0391, -0.9719),
        Atom("H", -1.9151, -0.0163, -1.9121),
        Atom("H", -1.6295, 0.6929, -0.2173),
        Atom("H", -0.9389, -0.9786, -0.6534),
    ]
    reac_shift.print_xyz_file()

    templates = get_ts_templates()
    assert len(templates) == 1
    assert templates[0].graph.number_of_nodes() == 6

    tsg_template = get_template_ts_guess(
        reac_shift,
        product_complex,
        name="template",
        bond_rearr=bond_rearr,
        method=XTB(),
    )

    # Reset the folder path to the default
    Config.ts_template_folder_path = None

    assert tsg_template is not None


@testutils.work_in_zipped_dir(os.path.join(here, "data", "ts_guess.zip"))
def test_ts_template_with_scan():

    if shutil.which("xtb") is None or not shutil.which("xtb").endswith("xtb"):
        return

    Config.lcode = "xtb"
    Config.XTB.path = shutil.which("xtb")

    ts_guess = TSguess(
        atoms=[
            Atom("C", -0.29102, 0.31489, -0.00001),
            Atom("Cl", 1.48694, 0.31490, 0.00000),
            Atom("H", -0.66083, -0.11622, -0.95298),
            Atom("H", -0.66086, 1.35574, 0.10314),
            Atom("H", -0.66083, -0.29487, 0.84984),
            Atom("Cl", -5.20436, 0.68301, -0.00000),
        ],
        charge=-1,
    )
    ts_guess.solvent = "water"

    # Running this constrained optimisation would break without intermediate
    # steps, so check that it worksw
    ts_guess.run_constrained_opt(
        "ll_const_opt", distance_consts={(0, 5): 2.3}, method=XTB()
    )

    # Ensure the correct final distance
    assert np.isclose(ts_guess.distance(0, 5), 2.3, atol=0.1)


@testutils.work_in_zipped_dir(os.path.join(here, "data", "ts_template.zip"))
def test_truncated_mol_graph_atom_types():

    ir_ts = TransitionState(
        TSguess(atoms=xyz_file_to_atoms("vaskas_TS.xyz"), charge=0, mult=1)
    )

    ir_ts.bond_rearrangement = BondRearrangement(
        forming_bonds=[(5, 1), (6, 1)], breaking_bonds=[(5, 6)]
    )
    if (5, 1) in ir_ts.graph.edges:
        ir_ts.graph.remove_edge(5, 1)
    ir_ts.graph.add_edge(5, 1, active=True)

    if (6, 1) in ir_ts.graph.edges:
        ir_ts.graph.remove_edge(6, 1)
    ir_ts.graph.add_edge(6, 1, active=True)

    if (5, 6) in ir_ts.graph.edges:
        ir_ts.graph.remove_edge(5, 6)
    ir_ts.graph.add_edge(5, 6, active=True)

    graph = get_truncated_active_mol_graph(ir_ts.graph)
    # Should only be a single Ir atom in the template
    assert (
        sum(node[1]["atom_label"] == "Ir" for node in graph.nodes(data=True))
        == 1
    )


def test_ts_template_parse():

    # No value
    with pytest.raises(TemplateLoadingFailed):
        _ = get_value_from_file("solvent", file_lines=["solvent:"])

    # Key doesn't exist
    with pytest.raises(TemplateLoadingFailed):
        _ = get_value_from_file("charge", file_lines=["solvent:"])
        _ = get_values_dict_from_file("charge", file_lines=["solvent:"])

    # Incorrectly formatted values section
    with pytest.raises(TemplateLoadingFailed):
        _ = get_values_dict_from_file("nodes", file_lines=["0 C", "1 F"])


@work_in_tmp_dir()
def test_ts_templates_find():

    templates = get_ts_templates(folder_path="/a/path/that/doesnt/exist")
    assert len(templates) == 0

    # Create a incorrectly formatted file, i.e. blank
    open("wrong_template.txt", "w").close()
    templates = get_ts_templates(folder_path=os.getcwd())
    assert len(templates) == 0

    os.remove("wrong_template.txt")


def test_inactive_graph():

    # Should fail to get a active graph from a graph with no active edges
    with pytest.raises(ValueError):
        _ = get_truncated_active_mol_graph(ch3f.graph)

    template = TStemplate()
    assert not template.graph_has_correct_structure()

    template.graph = ch3f.graph.copy()
    assert not template.graph_has_correct_structure()


@testutils.work_in_zipped_dir(os.path.join(here, "data", "ts_template.zip"))
def test_ts_from_species_is_same_as_from_ts_guess():

    ts = TransitionState(
        TSguess(atoms=xyz_file_to_atoms("vaskas_TS.xyz"), charge=0, mult=1)
    )

    assert TransitionState.from_species(Molecule("vaskas_TS.xyz")) == ts
