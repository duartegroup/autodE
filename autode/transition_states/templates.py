import os
import autode
from datetime import date
from autode.mol_graphs import MolecularGraph
from autode.config import Config
from autode.log import logger
from autode.mol_graphs import is_isomorphic
from autode.exceptions import TemplateLoadingFailed
from autode.solvent.solvents import get_solvent

"""
The idea with templating is to avoid needless PES scans when finding TSs for
which similar have already been found.

For instance the TS for the addition of CN- to acetone is going to be perturbed
only slightly by modifying a methyl for a CH2CH3 group. It is more efficient
to know the forming C–C bond distance in the previous TS, fix it and run
a constrained optimisation which will hopefully be a good guess of the TS
"""


def get_ts_template_folder_path(folder_path):
    """
    Get the full path to the directory containing the transition state
    templates, if it's unset then use the default folder in Config if it is
    set, or the autode/transition_states/lib folder where autodE is installed

    ---------------------------------------------------------------------------
    Arguments:
        folder_path: (str or None)

    Returns:
        (str): Path to the folder containing TS templates
    """

    if folder_path is not None:
        return folder_path

    logger.info("Folder path is not set – TS templates in the default path")

    if Config.ts_template_folder_path == "":
        raise ValueError(
            "Cannot set ts_template_folder_path to an empty string"
        )

    if Config.ts_template_folder_path is not None:
        logger.info("Configuration ts_template_folder_path is set")
        return Config.ts_template_folder_path

    else:
        ts_dir_path = os.path.dirname(os.path.abspath(__file__))
        return os.path.join(ts_dir_path, "lib")


def get_ts_templates(folder_path=None):
    """Get all the transition state templates from a folder, or the default if
    folder path is None. Transition state templates should be .txt files with
    at least a charge, multiplicity, solvent, and a graph with some active
    edge including distances.

    ---------------------------------------------------------------------------
    Keyword Arguments:
        folder_path (str): e.g. '/path/to/the/ts/template/library'

    Returns:
        (list(autode.transition_states.templates.TStemplate)): List of
        templates
    """
    folder_path = get_ts_template_folder_path(folder_path)
    logger.info(f"Getting TS templates from {folder_path}")

    if not os.path.exists(folder_path):
        logger.error("Folder does not exist")
        return []

    templates = []

    # Attempt to form transition state templates for all the .txt files in the
    # TS template folder
    for filename in os.listdir(folder_path):
        if not filename.endswith(".txt"):
            continue

        try:
            template = TStemplate(filename=os.path.join(folder_path, filename))
            templates.append(template)

        except TemplateLoadingFailed:
            logger.warning(f"Failed to load a template for {filename}")

    logger.info(f"Have {len(templates)} TS templates")
    return templates


def template_matches(reactant, truncated_graph, ts_template):
    r"""
    Determine if a transition state template matches a truncated graph. The
    truncated graph includes all the active bonds in the reaction and the
    nearest neighbours to those atoms e.g. for a Diels-Alder reaction::

             H        H
              \      /
            H-C----C-H          where the dotted lines represent active bonds
              .    .
          H  .      .   H
           \.        . /
        H - C        C - H
             \      /
              C    C

    where the full reaction is between ethene and butadiene.
    ---------------------------------------------------------------------------
    Arguments:
        reactant (autode.complex.ReactantComplex):

        truncated_graph (nx.Graph):

        ts_template (autode.transition_states.templates.TStemplate):

    Returns:
        (bool): Template matches
    """

    if (
        reactant.charge != ts_template.charge
        or reactant.mult != ts_template.mult
    ):
        return False

    if reactant.solvent != ts_template.solvent:
        return False

    if is_isomorphic(truncated_graph, ts_template.graph):
        logger.info("Found matching TS template")
        return True

    return False


def get_value_from_file(key, file_lines):
    """
    Get the value given a key from a list of file lines i.e. a saved template

    Example::

        Input:
        file_lines=
        _________________________
        .
        multiplicity: 1
        .
        ------------------------
        key='multiplicity'

        Output:
        1

    ---------------------------------------------------------------------------
    Arguments:
        key (str):
        file_lines (list(str)):

    Returns:
         (str): Value

    Raise:
        (autode.exceptions.TemplateLoadingFailed): If values not found
    """

    for i, line in enumerate(file_lines):
        if not line.startswith(str(key)):
            continue

        try:
            _, value = line.split()
            return value

        except (TypeError, ValueError):
            raise TemplateLoadingFailed(f"Incorrectly formatted line {i}")

    raise TemplateLoadingFailed(f"Did not find a {key} template")


def get_values_dict_from_file(key, file_lines):
    """
    Get the value given a key from a list of file lines i.e. a saved template.
    Example::

        Input:
        file_lines=
        _________________________
        .
        .
        multiplicity: 1
        nodes:
            0: atom_label=F
            2: atom_label=C
                 .
                 .
        ------------------------
        key='nodes'

        Output:
        {0: {'atom_label': 'F'}, 2: {'atom_label': 'C'}, ..}

    ---------------------------------------------------------------------------
    Arguments:
        key (str):
        file_lines (list(str)):

    Returns:
         (dict): Value

    Raise:
          (autode.exceptions.TemplateLoadingFailed): If values not found
    """
    key_lines = [line for line in file_lines if line.startswith(key)]

    if len(key_lines) != 1:
        raise TemplateLoadingFailed(f"Incorrect format of {key} section")

    values_dict = {}

    # Enumerate all indented lines starting after this key
    line_idx = file_lines.index(key_lines[0])

    for i, line in enumerate(file_lines[line_idx + 1 :]):
        # Only consider indented lines
        if not line.startswith("    "):
            break

        # Split the line on spaces
        items = line.split()

        if not items[0].endswith(":"):
            raise TemplateLoadingFailed(f"Key error on line {i}")

        # This key in the value dictionary is the first item in the line with
        # the whitespace and final colon removed
        v_key = items[0][:-1]

        # If the key is e.g. 0-1 as an edge then split it to the tuple (0, 1)
        if "-" in v_key:
            v_key = tuple(int(idx) for idx in v_key.split("-"))

        else:
            v_key = int(v_key)

        p_dict = {}

        # Expecting the remaining items to be separated by equals symbols
        # e.g. active=True
        for item in items[1:]:
            p_key, p_value = item.split("=")

            if p_value.lower() == "true":
                p_value = True

            elif p_value.lower() == "false":
                p_value = False

            else:
                try:
                    p_value = float(p_value)

                except ValueError:
                    pass

            p_dict[p_key] = p_value

        values_dict[v_key] = p_dict

    logger.info(f"Found {key}: {list(values_dict.keys())}")
    return values_dict


class TStemplate:
    def __init__(
        self,
        graph=None,
        charge=None,
        mult=None,
        solvent=None,
        species=None,
        filename=None,
    ):
        """
        TS template

        -----------------------------------------------------------------------
        Keyword Arguments:
            graph (nx.Graph): Active bonds in the TS are represented by the
                  edges with attribute active=True, going out to nearest bonded
                  neighbours

            solvent (autode.solvent.solvents.Solvent):

            charge (int):

            mult (int):

            species (autode.species.Species):

            filename (str): Saved template to load
        """

        self._filename = filename
        self.graph = graph
        self.solvent = solvent
        self.charge = charge
        self.mult = mult

        if species is not None:
            self.solvent = species.solvent
            self.charge = species.charge
            self.mult = species.mult

        if self._filename is not None:
            self.load(filename)

    def _save_to_file(self, file):
        """Save this template to a plain text .txt file with a ~yaml syntax"""

        title_line = (
            f"TS template generated by autode v.{autode.__version__}"
            f" on {date.today()}\n"
        )

        # Add nodes as a list, and their atom labels/symbols
        nodes_str = ""
        for i, data in self.graph.nodes(data=True):
            nodes_str += f'    {i}: atom_label={data["atom_label"]}\n'

        # Add edges as a list and their associated properties as a dict
        edges_str = ""
        for i, j, data in self.graph.edges(data=True):
            edge_str = f"    {i}-{j}: "

            if "pi" in data.keys():
                edge_str += f'pi={str(data["pi"])} '

            if "active" in data.keys():
                edge_str += f'active={str(data["active"])} '

            if "distance" in data.keys():
                edge_str += f'distance={data["distance"]:.4f} '

            edges_str += f"{edge_str}\n"

        print(
            title_line,
            f"solvent: {self.solvent}",
            f"charge: {self.charge}",
            f"multiplicity: {self.mult}",
            "nodes:",
            nodes_str,
            "edges:",
            edges_str,
            sep="\n",
            file=file,
        )

        return None

    def graph_has_correct_structure(self):
        """Check that the graph has some active edges and distances"""

        if self.graph is None:
            logger.warning("Incorrect TS template stricture - it was None!")
            return False

        n_active_edges = 0
        for edge in self.graph.edges:
            if "active" not in self.graph.edges[edge].keys():
                continue

            if not self.graph.edges[edge]["active"]:
                continue

            if (
                self.graph.edges[edge]["active"]
                and "distance" not in self.graph.edges[edge].keys()
            ):
                logger.warning("Active edge has no distance")
                return False

            n_active_edges += 1

        # A reasonably structured graph has at least 1 active edge
        if n_active_edges >= 1:
            return True

        else:
            logger.warning("Graph had no active edges")
            return False

    def save(self, basename="template", folder_path=None):
        """
        Save the TS template object in a plain text .txt file. With folder_path
        =None then the template will be saved to the default directory
        (see get_ts_template_folder_path). The name of the file will be
        basename.txt where i is an integer iterated until the file doesn't
        already exist.

        -----------------------------------------------------------------------
        Keyword Arguments:
            basename (str):

            folder_path (str or None):
        """

        folder_path = get_ts_template_folder_path(folder_path)
        logger.info(f"Saving TS template to {folder_path}")

        if not os.path.exists(folder_path):
            logger.info(f"Making directory {folder_path}")
            os.mkdir(folder_path)

        # Iterate i until the templatei.obj file doesn't exist
        name, i = basename + "0", 0
        while True:
            if not os.path.exists(os.path.join(folder_path, f"{name}.txt")):
                break
            name = basename + str(i)
            i += 1

        file_path = os.path.join(folder_path, f"{name}.txt")
        logger.info(f"Saving the template as {file_path}")

        with open(file_path, "w") as template_file:
            self._save_to_file(template_file)

        return None

    def load(self, filename):
        """
        Load a template from a saved file

        -----------------------------------------------------------------------
        Arguments:
            filename (str):

        Raise:
            (autode.exceptions.TemplateLoadingFailed):
        """
        try:
            template_lines = open(filename, "r").readlines()
        except (IOError, UnicodeDecodeError):
            raise TemplateLoadingFailed("Failed to read file lines")

        if len(template_lines) < 5:
            raise TemplateLoadingFailed("Not enough lines in the template")

        name = get_value_from_file("solvent", template_lines)

        if name.lower() == "none":
            self.solvent = None
        else:
            self.solvent = get_solvent(solvent_name=name, kind="implicit")

        self.charge = int(get_value_from_file("charge", template_lines))
        self.mult = int(get_value_from_file("multiplicity", template_lines))

        # Set the template graph by adding nodes and edges with atoms labels
        # and active/pi/distance attributes respectively
        self.graph = MolecularGraph()

        nodes = get_values_dict_from_file("nodes", template_lines)
        for idx, data in nodes.items():
            self.graph.add_node(idx, **data)

        edges = get_values_dict_from_file("edges", template_lines)

        for pair, data in edges.items():
            self.graph.add_edge(*pair, **data)

        if not self.graph_has_correct_structure():
            raise TemplateLoadingFailed("Incorrect graph structure")

        return None

    @property
    def filename(self) -> str:
        return "unknown" if self._filename is None else self._filename
