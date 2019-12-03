from autode.config import Config
from autode.log import logger
from rdkit.Chem import AllChem
from rdkit import Chem
import rdkit.Chem.Descriptors
from autode import mol_graphs
from autode.constants import Constants
from autode.conformers.conformers import generate_unique_rdkit_confs
from autode.bond_lengths import get_xyz_bond_list
from autode.bond_lengths import get_bond_list_from_rdkit_bonds
from autode.bond_lengths import get_avg_bond_length
from autode.bond_rearrangement import BondRearrangement
from autode.geom import calc_distance_matrix
from autode.geom import xyz2coord
from autode.conformers.conformers import extract_xyzs_from_rdkit_mol_object
from autode.conformers.conformers import Conformer
from autode.conformers.conformers import rdkit_conformer_geometries_are_resonable
from autode.conformers.conf_gen import gen_simanl_conf_xyzs
from autode.calculation import Calculation
from autode.methods import get_hmethod
import numpy as np


class Molecule:

    def _calc_multiplicity(self, n_radical_electrons):
        """
        Calculate the spin multiplicity 2S + 1 where S is the number of unpaired electrons
        :return:
        """
        if n_radical_electrons == 1:
            return 2

        if n_radical_electrons > 1:
            logger.warning(
                'Diradicals by default singlets. Set mol.mult if it\'s any different')

        return self.mult

    def _check_rdkit_graph_agreement(self):
        try:
            assert self.n_bonds == self.graph.number_of_edges()
        except AssertionError:
            logger.error(
                'Number of rdkit bonds doesn\'t match the the molecular graph')
            exit()

    def _find_pi_systems(self):
        logger.info('Finding pi systems')
        pi_systems = []
        pi_system = None
        if self.pi_bonds is not None:
            for pi_bond in self.pi_bonds:
                for system in pi_systems:
                    if pi_bond in system:
                        pi_system = system
                        pi_systems.remove(pi_system)
                        break
                if pi_system is None:
                    pi_system = []
                for other_pi_bond in self.pi_bonds:
                    if not other_pi_bond in pi_system:
                        for atom in other_pi_bond:
                            bonded_list = self.get_bonded_atoms_to_i(atom)
                            if any(bonded_atom in pi_bond for bonded_atom in bonded_list):
                                pi_system.append(other_pi_bond)
                pi_systems.append(pi_system)
                pi_system = None
        flat_pi_systems = []
        for system in pi_systems:
            flat_system = [atom for bond in system for atom in bond]
            flat_system_set = set(flat_system)
            flat_pi_systems.append(sorted(flat_system_set))
        self.pi_systems = flat_pi_systems

    def get_core_atoms(self, product_graph=None, depth=3):
        if self.active_atoms == None:
            logger.error('No active atoms found')
            return None

        core_atoms = set(self.active_atoms)
        for _ in range(depth-1):
            new_core_atoms = set()
            for atom in core_atoms:
                bonded_list = self.get_bonded_atoms_to_i(atom)
                for bonded_atom in bonded_list:
                    new_core_atoms.add(bonded_atom)
            core_atoms.update(new_core_atoms)

        old_core_atoms = set()

        cycles = mol_graphs.find_cycle(self.graph)
        if product_graph is not None:
            prod_cycles = cycle = mol_graphs.find_cycle(product_graph)

        while len(old_core_atoms) < len(core_atoms):
            old_core_atoms = core_atoms.copy()
            ring_atoms = set()
            logger.info('Looking for rings in the reactants')
            for atom in core_atoms:
                for cycle in cycles:
                    if atom in cycle:
                        for cycle_atom in cycle:
                            ring_atoms.add(cycle_atom)
            core_atoms.update(ring_atoms)

            logger.info('Looking for rings in the products')
            if product_graph is None:
                logger.warning(
                    'No product graph found, this will cause errors if rings are formed in the reaction')
            else:
                prod_ring_atoms = set()
                for atom in core_atoms:
                    for cycle in prod_cycles:
                        if atom in cycle:
                            for cycle_atom in cycle:
                                prod_ring_atoms.add(cycle_atom)
                core_atoms.update(prod_ring_atoms)

            if self.pi_systems is not None:
                logger.info('Checking for pi bonds')
                core_atoms_pi_bonds = set()
                for atom in core_atoms:
                    for system in self.pi_systems:
                        if atom in system:
                            for other_atom in system:
                                core_atoms_pi_bonds.add(other_atom)
                            break
                core_atoms.update(core_atoms_pi_bonds)

            # don't want to make OH, SH or NH, as these can be acidic and can mess things up
            bonded_to_heteroatoms = set()
            for atom in core_atoms:
                if self.get_atom_label(atom) in ('O', 'S', 'N'):
                    bonded_atoms = self.get_bonded_atoms_to_i(atom)
                    for bonded_atom in bonded_atoms:
                        bonded_to_heteroatoms.add(bonded_atom)
            core_atoms.update(bonded_to_heteroatoms)

        core_atoms_no_other_bonded = set()
        for atom in core_atoms:
            bonded_list = self.get_bonded_atoms_to_i(atom)
            for bonded_atom in bonded_list:
                no_bonded_atoms = len(self.get_bonded_atoms_to_i(bonded_atom))
                if no_bonded_atoms == 1:
                    core_atoms_no_other_bonded.add(bonded_atom)
        core_atoms.update(core_atoms_no_other_bonded)

        return sorted(core_atoms)

    def calc_bond_distance(self, bond):
        return self.distance_matrix[bond[0], bond[1]]

    def get_possible_forming_bonds(self):
        curr_bonds = [pair for pair in self.graph.edges()]
        return [(i, j) for i in range(self.n_atoms) for j in range(self.n_atoms)
                if i < j and (i, j) not in curr_bonds and (j, i) not in curr_bonds]

    def get_possible_breaking_bonds(self):
        return [pair for pair in self.graph.edges()]

    def get_atom_label(self, atom_i):
        return self.xyzs[atom_i][0]

    def get_bonded_atoms_to_i(self, atom_i):
        bonded_atoms = []
        for edge in self.graph.edges():
            if edge[0] == atom_i:
                bonded_atoms.append(edge[1])
            if edge[1] == atom_i:
                bonded_atoms.append(edge[0])
        return bonded_atoms

    def get_coords(self):
        return xyz2coord(self.xyzs)

    def get_active_mol_graph(self, active_bonds):
        logger.info('Getting molecular graph with active edges')
        active_graph = self.graph.copy()

        for bond in active_bonds:
            has_edge = False
            for edge in active_graph.edges():
                if edge == bond or edge == tuple(reversed(bond)):
                    has_edge = True
                    active_graph.edges[edge[0], edge[1]]['active'] = True

            if not has_edge:
                active_graph.add_edge(*bond, active=True)

        return active_graph

    def generate_conformers(self, n_rdkit_confs=300):

        self.conformers = []

        if self.rdkit_conf_gen_is_fine:
            logger.info(
                'Generating Molecule conformer xyz lists from rdkit mol object')
            unique_conf_ids = generate_unique_rdkit_confs(
                self.mol_obj, n_rdkit_confs)
            logger.info(
                f'Generated {len(unique_conf_ids)} unique conformers with RDKit ETKDG')
            conf_xyzs = extract_xyzs_from_rdkit_mol_object(
                self, conf_ids=unique_conf_ids)

        else:
            bond_list = get_bond_list_from_rdkit_bonds(
                rdkit_bonds_obj=self.mol_obj.GetBonds())
            conf_xyzs = gen_simanl_conf_xyzs(name=self.name, init_xyzs=self.xyzs, bond_list=bond_list,
                                             charge=self.charge, stereocentres=self.stereocentres)

        for i in range(len(conf_xyzs)):
            self.conformers.append(Conformer(name=self.name + '_conf' + str(i), xyzs=conf_xyzs[i],
                                             solvent=self.solvent, charge=self.charge, mult=self.mult))

        self.n_conformers = len(self.conformers)

    def strip_non_unique_confs(self, energy_threshold_kj=1):
        logger.info(
            'Stripping conformers with energy ∆E < 1 kJ mol-1 to others')
        # conformer.energy is in Hartrees
        d_e = energy_threshold_kj / Constants.ha2kJmol
        # The first conformer must be unique
        unique_conformers = [self.conformers[0]]

        for i in range(1, self.n_conformers):
            unique = True
            for j in range(len(unique_conformers)):
                if self.conformers[i].energy - d_e < self.conformers[j].energy < self.conformers[i].energy + d_e:
                    unique = False
                    break
            if unique:
                unique_conformers.append(self.conformers[i])

        logger.info(
            f'Stripped {self.n_conformers - len(unique_conformers)} conformers from a total of {self.n_conformers}')
        self.conformers = unique_conformers
        self.n_conformers = len(self.conformers)

    def strip_core(self, core_atoms, bond_rearrang=None):
        logger.info('Stripping the extraneous atoms')
        bonded_to_core = set()
        bond_from_core = []

        if core_atoms is None:
            logger.info('No core atoms, not stripping extraneous atoms')
            return (self, bond_rearrang)

        if self.n_atoms - len(core_atoms) < 5:
            logger.info('Not enough atoms to strip for it to be worthwhile')
            return (self, bond_rearrang)

        coords = self.get_coords()

        # get the xyzs of the core part of the fragment
        for atom in core_atoms:
            bonded_atoms = self.get_bonded_atoms_to_i(atom)
            for bonded_atom in bonded_atoms:
                if not bonded_atom in core_atoms:
                    bonded_to_core.add(bonded_atom)
                    bond_from_core.append((atom, bonded_atom))
        fragment_xyzs = [self.xyzs[i] for i in core_atoms]

        # put the extra H's on the fragment
        for bond in bond_from_core:
            core_atom, bonded_atom = bond
            bond_vector = coords[bonded_atom] - coords[core_atom]
            normed_bond_vector = bond_vector / np.linalg.norm(bond_vector)
            avg_bond_length = get_avg_bond_length(
                self.get_atom_label(core_atom), 'H')
            h_coords = (coords[core_atom] +
                        normed_bond_vector * avg_bond_length).tolist()
            fragment_xyzs.append(['H'] + h_coords)

        if self.xyzs == fragment_xyzs:
            logger.info('No atoms to strip')
            return (self, bond_rearrang)

        if bond_rearrang is not None:
            # get the bond rearrangement in the new atom indices
            new_fbonds = []
            for fbond in bond_rearrang.fbonds:
                new_atom_1 = core_atoms.index(fbond[0])
                new_atom_2 = core_atoms.index(fbond[1])
                new_fbonds.append((new_atom_1, new_atom_2))
            new_bbonds = []
            for bbond in bond_rearrang.bbonds:
                new_atom_1 = core_atoms.index(bbond[0])
                new_atom_2 = core_atoms.index(bbond[1])
                new_bbonds.append((new_atom_1, new_atom_2))

            new_bond_rearrang = BondRearrangement(
                forming_bonds=new_fbonds, breaking_bonds=new_bbonds)
        else:
            new_bond_rearrang = None

        fragment = Molecule(name=self.name + '_fragment', xyzs=fragment_xyzs,
                            solvent=self.solvent, charge=self.charge, mult=self.mult, is_fragment=True)
        return (fragment, new_bond_rearrang)

    def find_lowest_energy_conformer(self):
        """
        For a molecule object find the lowest in energy and set it as the mol.xyzs and mol.energy
        :return:
        """
        self.generate_conformers()
        [self.conformers[i].optimise() for i in range(len(self.conformers))]
        self.strip_non_unique_confs()
        [self.conformers[i].optimise(method=self.method)
         for i in range(len(self.conformers))]

        lowest_energy = None
        for conformer in self.conformers:
            if conformer.energy is None:
                continue

            conformer_graph = mol_graphs.make_graph(
                conformer.xyzs, self.n_atoms)

            if mol_graphs.is_isomorphic(self.graph, conformer_graph):
                # If the conformer retains the same connectivity
                if lowest_energy is None:
                    lowest_energy = conformer.energy

                elif conformer.energy <= lowest_energy:
                    self.energy = conformer.energy
                    self.set_xyzs(conformer.xyzs)
                    lowest_energy = conformer.energy

                else:
                    pass
            else:
                logger.warning(
                    'Conformer had a different molecular graph. Ignoring')

        logger.info(
            'Set lowest energy conformer energy & geometry as mol.energy & mol.xyzs')

    def set_xyzs(self, xyzs):
        logger.info('Setting molecule xyzs')
        self.xyzs = xyzs
        if xyzs is None:
            logger.error('Setting xyzs as None')
            return

        self.n_atoms = len(xyzs)
        self.distance_matrix = calc_distance_matrix(xyzs)
        self.graph = mol_graphs.make_graph(xyzs, n_atoms=self.n_atoms)
        self.n_bonds = self.graph.number_of_edges()

    def optimise(self, method=None):
        logger.info(f'Running optimisation of {self.name}')
        if method is None:
            method = self.method

        opt = Calculation(name=self.name + '_opt', molecule=self, method=method, keywords=method.opt_keywords,
                          n_cores=Config.n_cores, opt=True, max_core_mb=Config.max_core)
        opt.run()
        self.energy = opt.get_energy()
        self.set_xyzs(xyzs=opt.get_final_xyzs())
        self.pi_bonds = opt.get_pi_bonds()
        if self.pi_bonds is not None:
            self._find_pi_systems()

    def single_point(self, method=None):
        logger.info(f'Running single point energy evaluation of {self.name}')
        if method is None:
            method = self.method

        sp = Calculation(name=self.name + '_sp', molecule=self, method=method, keywords=method.sp_keywords,
                         n_cores=Config.n_cores, max_core_mb=Config.max_core)
        sp.run()
        self.energy = sp.get_energy()

    def _init_smiles(self, name, smiles):

        try:
            self.mol_obj = Chem.MolFromSmiles(smiles)
            self.mol_obj = Chem.AddHs(self.mol_obj)
        except RuntimeError:
            logger.critical(
                f'Could not generate an rdkit mol object for {name}')
            exit()

        self.n_atoms = self.mol_obj.GetNumAtoms()
        self.n_bonds = len(self.mol_obj.GetBonds())
        self.charge = Chem.GetFormalCharge(self.mol_obj)
        n_radical_electrons = rdkit.Chem.Descriptors.NumRadicalElectrons(
            self.mol_obj)
        self.mult = self._calc_multiplicity(n_radical_electrons)

        AllChem.EmbedMultipleConfs(
            self.mol_obj, numConfs=1, params=AllChem.ETKDG())
        self.xyzs = extract_xyzs_from_rdkit_mol_object(self, conf_ids=[0])[0]

        unassigned_stereocentres = False
        stereocentres = []
        stereochem_info = Chem.FindMolChiralCenters(
            self.mol_obj, includeUnassigned=True)
        for atom, assignment in stereochem_info:
            if assignment == '?':
                unassigned_stereocentres = True
            else:
                stereocentres.append(atom)

        if len(stereocentres) > 0:
            self.stereocentres = stereocentres
            if unassigned_stereocentres:
                logger.warning('Unassigned stereocentres found')

        if not rdkit_conformer_geometries_are_resonable(conf_xyzs=[self.xyzs]):
            logger.info('RDKit conformer was not reasonable')
            self.rdkit_conf_gen_is_fine = False
            bond_list = get_bond_list_from_rdkit_bonds(
                rdkit_bonds_obj=self.mol_obj.GetBonds())
            self.xyzs = gen_simanl_conf_xyzs(name=self.name, init_xyzs=self.xyzs, bond_list=bond_list,
                                             charge=self.charge, stereocentres=self.stereocentres, n_simanls=1)[0]
            self.graph = mol_graphs.make_graph(self.xyzs, self.n_atoms)
            self.n_bonds = self.graph.number_of_edges()

        else:
            self.graph = mol_graphs.make_graph(self.xyzs, self.n_atoms)
            self._check_rdkit_graph_agreement()

        self.distance_matrix = calc_distance_matrix(self.xyzs)

    def _init_xyzs(self, xyzs):
        for xyz in xyzs:
            if len(xyz) != 4:
                logger.critical(
                    f'XYZ input is not the correct format (needs to be e.g. [\'H\',0,0,0]). Found {xyz} instead')
                exit()

        if isinstance(self, (Reactant, Product)):
            logger.warning(
                'Initiating a molecule from xyzs means any stereocentres will probably be lost. Initiate from a SMILES string to keep stereochemistry')

        self.n_atoms = len(xyzs)
        self.n_bonds = len(get_xyz_bond_list(xyzs))
        self.graph = mol_graphs.make_graph(self.xyzs, self.n_atoms)
        self.distance_matrix = calc_distance_matrix(xyzs)

    def __init__(self, name='molecule', smiles=None, xyzs=None, solvent=None, charge=0, mult=1, is_fragment=False):
        """
        Initialise a Molecule object.
        Will generate xyz lists of all the conformers found by RDKit within the number
        of conformers searched (n_confs)
        :param name: (str) Name of the molecule
        :param smiles: (str) Standard SMILES string. e.g. generated by Chemdraw
        :param solvent: (str) Solvent that the molecule is immersed in. Will be used in optimise() and single_point()
        :param charge: (int) charge on the molecule
        :param mult: (int) spin multiplicity on the molecule
        """
        logger.info(f'Generating a Molecule object for {name}')

        self.name = name
        self.smiles = smiles
        self.solvent = solvent
        self.charge = charge
        self.mult = mult
        self.xyzs = xyzs
        self.is_fragment = is_fragment
        self.method = get_hmethod()
        self.mol_obj = None
        self.energy = None
        self.n_atoms = None
        self.n_bonds = None
        self.conformers = None
        self.n_conformers = None
        self.rdkit_conf_gen_is_fine = True
        self.graph = None
        self.distance_matrix = None
        self.active_atoms = None
        self.stereocentres = None
        self.pi_bonds = None
        self.pi_systems = None

        if smiles:
            self._init_smiles(name, smiles)

        if xyzs:
            self._init_xyzs(xyzs)


class Reactant(Molecule):
    pass


class Product(Molecule):
    pass
