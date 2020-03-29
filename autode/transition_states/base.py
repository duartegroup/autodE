import numpy as np
from copy import deepcopy
from autode.species import Species
from autode.methods import get_hmethod
from autode.geom import length
from autode.config import Config
from autode import mol_graphs
from autode.mol_graphs import is_isomorphic
from autode.calculation import Calculation
from autode.exceptions import NoNormalModesFound
from autode.exceptions import AtomsNotFound
from autode.exceptions import NoCalculationOutput
from autode.log import logger
from autode.atoms import get_atomic_weight


class TSbase(Species):

    def _init_graph(self):
        """Set the molecular graph for this TS object from the reactant"""
        logger.warning(f'Setting the graph of {self.name} from reactants')

        self.graph = self.reactant.graph.copy()
        return None

    def could_have_correct_imag_mode(self, method=None, threshold=-50):
        """
        Determine if a point on the PES could have the correct imaginary mode. This must have

        (0) An imaginary frequency      (quoted as negative in most EST codes)
        (1) The most negative(/imaginary) is more negative that a threshold

        Keywords Arguments:
            calc (autode.calculation.Calculation):
            method (autode.wrappers.base.ElectronicStructureMethod):
            threshold (float):
        """

        if self.calc is None:
            logger.info('Calculating the hessian ')
            self.calc = Calculation(name=self.name + '_hess', molecule=self, method=method,
                                    keywords_list=method.keywords.hess, n_cores=Config.n_cores)
            self.calc.run()

        imag_freqs = self.calc.get_imag_freqs()

        if len(imag_freqs) == 0:
            logger.warning('Hessian had no imaginary modes')
            return False

        if len(imag_freqs) > 1:
            logger.warning(f'Hessian had {len(imag_freqs)} imaginary modes')

        if imag_freqs[0] > threshold:
            logger.warning('Imaginary modes were too small to be significant')
            return False

        logger.info('Species could have the correct imaginary mode')
        return True

    def has_correct_imag_mode(self, active_atoms, calc=None, method=None, ensure_links=False):
        """Check that the imaginary mode is 'correct' set the calculation (hessian or optts)"""
        self.calc = calc

        # By default the high level method is used to check imaginary modes
        if method is None:
            method = get_hmethod()

        # Run a fast check on  whether it's likely the mode is correct
        if not self.could_have_correct_imag_mode(method=method):
            return False

        if not ts_has_contribution_from_active_atoms(calc=self.calc, active_atoms=active_atoms):
            logger.info('Species does not have the correct imaginary mode')
            return False

        # If requested, perform displacements over the imaginary mode to ensure the mode connects reactants and products
        if ensure_links:

            if imag_mode_links_reactant_products(calc, self.reactant.graph, self.product.graph, method=method):
                logger.info('Imaginary mode links reactants and products, TS found')
                return True

            else:
                logger.warning('Imaginary mode does not link reactants and products, TS *not* found')
                return False

        logger.warning('Species may have the correct imaginary mode')
        return True

    def __init__(self, atoms, reactant, product, name='ts_guess'):

        super().__init__(name=name, atoms=atoms, charge=reactant.charge, mult=reactant.mult)
        self.solvent = reactant.solvent
        self.atoms = atoms

        self.reactant = reactant
        self.product = product

        self.calc = None

        self._init_graph()


def ts_has_contribution_from_active_atoms(calc, active_atoms, threshold=0.15):
    """For a hessian calculation check that the first imaginary mode (number 6) in the final frequency calculation
    contains the correct motion, i.e. contributes more than threshold_contribution in relative terms to the
    magnitude of the sum of the forces

    Arguments:
        calc (autode.calculation.Calculation): calculation object
        active_atoms (list(int)):

    Keyword Arguments:
        threshold (float): threshold contribution to the imaginary mode from the atoms in

    Returns:
        (bool): if the imaginary mode is correct or not
    """
    logger.info(f'Checking the active atoms contribute more than {threshold} to the imag mode')

    try:
        imag_normal_mode_displacements_xyz = calc.get_normal_mode_displacements(mode_number=6)

    except (NoNormalModesFound, NoCalculationOutput):
        logger.error('Have no imaginary normal mode displacements to analyse')
        return False

    # If there are normal modes then there should be atoms in the output..
    atoms = calc.get_final_atoms()

    # Calculate the magnitudes of the motion on each atom weighted by the atomic weight
    imag_mode_magnitudes = [length(np.array(dis_xyz)) for dis_xyz in imag_normal_mode_displacements_xyz]
    weighted_imag_mode_magnitudes = [get_atomic_weight(atom_label=atoms[i].label) + 10 * imag_mode_magnitudes[i]
                                     for i in range(len(atoms))]

    # Calculate the sum of the weighted magnitudes on the active atoms
    sum_active_atom_magnitudes = sum([weighted_imag_mode_magnitudes[atom_index] for atom_index in active_atoms])

    rel_contribution = sum_active_atom_magnitudes / np.sum(np.array(weighted_imag_mode_magnitudes))

    if rel_contribution > threshold:
        logger.info(f'Significant contribution from active atoms to imag mode. (contribution = {rel_contribution:.3f})')
        return True

    else:
        logger.warning(f'TS has *no* significant contribution from the active atoms to the imag mode '
                       f'(contribution = {rel_contribution:.3f})')
        return False


def get_displaced_atoms_along_mode(calc, mode_number, disp_magnitude=1.0):
    """Displace the geometry along the imaginary mode with mode number iterating from 0, where 0-2 are translational
    normal modes, 3-5 are rotational modes and 6 is the largest imaginary mode. To displace along the second imaginary
    mode we have mode_number=7

    Arguments:
        calc (autode.calculation.Calculation):
        mode_number (int): Mode number to displace along

    Keyword Arguments:
        disp_magnitude (float): Distance to displace (default: {1.0})

    Returns:
        (list(autode.atoms.Atom)):
    """
    logger.info('Displacing along imaginary mode')

    atoms = deepcopy(calc.get_final_atoms())
    mode_disp_coords = calc.get_normal_mode_displacements(mode_number=mode_number)      # n x 3 array

    assert len(atoms) == len(mode_disp_coords)

    for i in range(len(atoms)):
        atoms[i].translate(vec=disp_magnitude * mode_disp_coords[i, :])

    return atoms


def imag_mode_links_reactant_products(calc, reactant_graph, product_graph, method, disp_mag=1):
    """Displaces atoms along the imaginary mode forwards (f) and backwards (b) to see if products and reactants are made

    Arguments:
        calc (autode.calculation.Calculation):
        reactant_graph (networkx.Graph):
        product_graph (networkx.Graph):
        method (autode.wrappers.base.ElectronicStructureMethod):


    Keyword Arguments:
        disp_mag (int): Distance to be displaced along the imag mode (default: {1})

    Returns:
        bool: if the imag mode is correct or not
    """
    logger.info('Displacing along imag modes to check that the TS links reactants and products')

    # Get the species that is optimised by displacing forwards along the imaginary mode
    f_displaced_atoms = get_displaced_atoms_along_mode(calc, mode_number=6, disp_magnitude=disp_mag)
    f_displaced_mol = get_optimised_species(calc, method, direction='forwards', atoms=f_displaced_atoms)

    if not is_isomorphic(f_displaced_mol.graph, reactant_graph) and not is_isomorphic(f_displaced_mol.graph, product_graph):
        logger.warning('Forward displacement does not afford reactants or products')
        return False

    # Get the species that is optimised by displacing backwards along the imaginary mode
    b_displaced_atoms = get_displaced_atoms_along_mode(calc, mode_number=6, disp_magnitude=-disp_mag)
    b_displaced_mol = get_optimised_species(calc, method, direction='backwards', atoms=b_displaced_atoms)

    if any(mol.atoms is None for mol in (f_displaced_mol, b_displaced_mol)):
        logger.warning('Atoms not set in the output. Cannot calculate isomorphisms')
        return False

    if is_isomorphic(b_displaced_mol.graph, reactant_graph) and is_isomorphic(f_displaced_mol.graph, product_graph):
        logger.info('Forwards displacement lead to products and backwards')
        return True

    if is_isomorphic(f_displaced_mol.graph, reactant_graph) and is_isomorphic(b_displaced_mol.graph, product_graph):
        logger.info('Backwards displacement lead to products and forwards to reactants')
        return True
    
    # TODO check that there is an isomorphism up to a single weak interaction

    return False


def get_optimised_species(calc, method, direction, atoms):
    """Get the species that is optimised from an initial set of atoms"""

    species = Species(name=f'{calc.name}_{direction}', atoms=atoms, charge=calc.molecule.charge, mult=calc.molecule.mult)

    # Note that for the surface to be the same the keywords.opt and keywords.hess need to match in the level of theory
    calc = Calculation(name=f'{calc.name}_{direction}', molecule=species, method=method,
                       keywords_list=method.keywords.opt, n_cores=Config.n_cores, opt=True)
    calc.run()

    try:
        species.set_atoms(atoms=calc.get_final_atoms())
        species.energy = calc.get_energy()
        mol_graphs.make_graph(species)

    except AtomsNotFound:
        logger.error(f'{direction} displacement calculation failed')

    return species
