from autode.config import Config
from autode.log import logger
from rdkit.Chem import AllChem
from rdkit import Chem
import rdkit.Chem.Descriptors
from autode import mol_graphs
from autode.constants import Constants
from autode.mol_graphs import is_isomorphic
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
from autode.solvent.solvents import get_solvent
import numpy as np

from autode.solvent.explicit_solvent import do_explicit_solvent_qmmm


class Molecule:

    def _calc_multiplicity(self, n_radical_electrons):
        """Calculate the spin multiplicity 2S + 1 where S is the number of unpaired electrons

        Arguments:
            n_radical_electrons (int): number of radical electrons

        Returns:
            int: multiplicity of the molecule
        """
        if n_radical_electrons == 1:
            return 2

        if n_radical_electrons > 1:
            logger.warning('Diradicals by default singlets. Set mol.mult if it\'s any different')

        return self.mult

    def _check_rdkit_graph_agreement(self):
        try:
            assert self.n_bonds == self.graph.number_of_edges()
        except AssertionError:
            logger.error('Number of rdkit bonds doesn\'t match the the molecular graph')
            exit()

    def set_solvent(self, solvent):
        if isinstance(solvent, str):
            self.solvent = get_solvent(solvent)
            if self.solvent is None:
                logger.critical('Could not find the solvent specified')
                exit()
        else:
            self.solvent = solvent

    def get_core_atoms(self, product_graph, depth=3):
        """Finds the 'core' of a molecule, to find atoms that should not be stripped. Core atoms are those within a certain
        number of bonds from atoms that are reacting. Also checks to ensure rings and pi bonda aren't being broken.

        Arguments"
            product_graph (nx.Graph): Graph of the product molecule (with the atom indices of the reactants) to see if a ring is being made

        Keyword Arguments:
            depth (int): Number of bonds from the active atoms within which atoms are 'core' (default: {3})

        Returns:
            list: list of core atoms
        """
        if self.active_atoms is None:
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

        reac_cycles = mol_graphs.find_cycles(self.graph)
        prod_cycles = mol_graphs.find_cycles(product_graph)

        for prod_cycle in prod_cycles:
            match = False
            for reac_cycle in reac_cycles:
                if len(prod_cycle) == len(reac_cycle):
                    if all(prod_atom in reac_cycle for prod_atom in prod_cycle):
                        match = True
                        break
            if not match:
                for atom in prod_cycle:
                    core_atoms.add(atom)

        while len(old_core_atoms) < len(core_atoms):
            old_core_atoms = core_atoms.copy()
            logger.info('Looking for rings in the reactants')
            for i, cycle in enumerate(reac_cycles):
                atoms_in_ring = []
                for atom in core_atoms:
                    if atom in cycle:
                        atoms_in_ring.append(atom)
                add = False
                if len(atoms_in_ring) > 0 and len(cycle) == 3:
                    add = True
                elif len(atoms_in_ring) == 2:
                    already_in_ring = False
                    for j, other_cycle in enumerate(reac_cycles):
                        if i != j:
                            if atoms_in_ring[0] in other_cycle and atoms_in_ring[1] in other_cycle:
                                if all(atom in core_atoms for atom in other_cycle):
                                    already_in_ring = True
                                    break
                    if not already_in_ring:
                        add = True
                elif len(atoms_in_ring) > 2:
                    add = True
                if add:
                    for cycle_atom in cycle:
                        core_atoms.add(cycle_atom)

            if self.pi_bonds is not None:
                pi_core_atoms = set()
                logger.info('Checking for pi bonds')
                for atom in core_atoms:
                    for bond in self.pi_bonds:
                        if atom in bond:
                            for other_atom in bond:
                                pi_core_atoms.add(other_atom)
                            break
                core_atoms.update(pi_core_atoms)

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
            logger.info('Generating Molecule conformer xyz lists from rdkit mol object')
            unique_conf_ids = generate_unique_rdkit_confs(self.mol_obj, n_rdkit_confs)
            logger.info(f'Generated {len(unique_conf_ids)} unique conformers with RDKit ETKDG')
            conf_xyzs = extract_xyzs_from_rdkit_mol_object(self, conf_ids=unique_conf_ids)

        else:
            bond_list = get_bond_list_from_rdkit_bonds(rdkit_bonds_obj=self.mol_obj.GetBonds())
            conf_xyzs = gen_simanl_conf_xyzs(name=self.name, init_xyzs=self.xyzs, bond_list=bond_list, stereocentres=self.stereocentres)

        for i in range(len(conf_xyzs)):
            self.conformers.append(Conformer(name=self.name + '_conf' + str(i), xyzs=conf_xyzs[i],
                                             solvent=self.solvent, charge=self.charge, mult=self.mult))

        self.n_conformers = len(self.conformers)

    def strip_non_unique_confs(self, energy_threshold_kj=1):
        logger.info('Stripping conformers with energy ∆E < 1 kJ mol-1 to others')
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

        logger.info(f'Stripped {self.n_conformers - len(unique_conformers)} conformers from a total of {self.n_conformers}')
        self.conformers = unique_conformers
        self.n_conformers = len(self.conformers)

    def strip_core(self, core_atoms, bond_rearrang=None):
        """Removes extraneous atoms from the core, to hasten the TS search calculations. If given, it will also find the bond
        rearrangement in the new indices of the molecule. If there are too few atoms to strip it will return the original molecule.

        Arguments:
            core_atoms (list): list of core atoms, which must be kept

        Keyword Arguments:
            bond_rearrang (bond rearrang object): the bond rearrangement of the reaction (default: {None})

        Returns:
            tuple: (stripped molecule, new bond rearrangement)
        """
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
            avg_bond_length = get_avg_bond_length(self.get_atom_label(core_atom), 'H')
            h_coords = (coords[core_atom] +
                        normed_bond_vector * avg_bond_length).tolist()
            fragment_xyzs.append(['H'] + h_coords)

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

            new_bond_rearrang = BondRearrangement(forming_bonds=new_fbonds, breaking_bonds=new_bbonds)
        else:
            new_bond_rearrang = None

        fragment = Molecule(name=self.name + '_fragment', xyzs=fragment_xyzs,
                            solvent=self.solvent, charge=self.charge, mult=self.mult, is_fragment=True)
        return (fragment, new_bond_rearrang)

    def find_lowest_energy_conformer(self):
        """For a molecule object find the lowest conformer in energy and set it as the mol.xyzs and mol.energy
        """
        self.generate_conformers()
        [self.conformers[i].optimise() for i in range(len(self.conformers))]
        self.strip_non_unique_confs()
        [self.conformers[i].optimise(method=self.method) for i in range(len(self.conformers))]

        lowest_energy = None
        for conformer in self.conformers:
            if conformer.energy is None:
                continue

            conformer_graph = mol_graphs.make_graph(conformer.xyzs, self.n_atoms)

            if mol_graphs.is_isomorphic(self.graph, conformer_graph):
                # If the conformer retains the same connectivity
                if lowest_energy is None:
                    lowest_energy = conformer.energy

                if conformer.energy <= lowest_energy:
                    self.energy = conformer.energy
                    self.set_xyzs(conformer.xyzs)
                    self.charges = conformer.charges
                    lowest_energy = conformer.energy

                else:
                    pass
            else:
                logger.warning('Conformer had a different molecular graph. Ignoring')

        logger.info('Set lowest energy conformer energy & geometry as mol.energy & mol.xyzs')

    def set_xyzs(self, xyzs):
        logger.info('Setting molecule xyzs')
        self.xyzs = xyzs

        if xyzs is None:
            logger.error('Setting xyzs as None')
            return

        self.n_atoms = len(xyzs)
        self.distance_matrix = calc_distance_matrix(xyzs)

        old_xyzs_graph = self.graph.copy()
        self.graph = mol_graphs.make_graph(xyzs, n_atoms=self.n_atoms)

        if not is_isomorphic(old_xyzs_graph, self.graph):
            logger.warning('New xyzs result in a modified molecular graph')

        self.n_bonds = self.graph.number_of_edges()

    def optimise(self, solvent_mol, method=None):
        logger.info(f'Running optimisation of {self.name}')
        if method is None:
            method = self.method

        opt = Calculation(name=self.name + '_opt', molecule=self, method=method, keywords=method.opt_keywords,
                          n_cores=Config.n_cores, opt=True, max_core_mb=Config.max_core)
        opt.run()
        self.energy = opt.get_energy()
        self.set_xyzs(xyzs=opt.get_final_xyzs())
        self.charges = opt.get_atomic_charges()

        if solvent_mol is not None:
            _, qmmm_xyzs, n_qm_atoms = do_explicit_solvent_qmmm(self, solvent_mol, method, n_qm_solvent_mols=30)
            self.xyzs = qmmm_xyzs[:self.n_atoms]
            self.qm_solvent_xyzs = qmmm_xyzs[self.n_atoms: n_qm_atoms]
            self.mm_solvent_xyzs = qmmm_xyzs[n_qm_atoms:]

    def single_point(self, solvent_mol, method=None):
        logger.info(f'Running single point energy evaluation of {self.name}')
        if method is None:
            method = self.method

        if solvent_mol:
            point_charges = []
        for i, xyz in enumerate(self.mm_solvent_xyzs):
            point_charges.append(xyz + [solvent_mol.charges[i % solvent_mol.n_atoms]])
        else:
            point_charges = None

        sp = Calculation(name=self.name + '_sp', molecule=self, method=method, keywords=method.sp_keywords,
                         n_cores=Config.n_cores, max_core_mb=Config.max_core, charges=point_charges)
        sp.run()
        self.energy = sp.get_energy()

    def is_possible_pi_atom(self, atom_i):
        # metals may break this
        pi_valencies = {'B': [1, 2], 'N': [1, 2], 'O': [1], 'C': [1, 2, 3], 'P': [1, 2, 3, 4], 'S': [1, 3, 4, 5], 'Si': [1, 2, 3]}
        atom_label = self.get_atom_label(atom_i)
        if atom_label in pi_valencies.keys():
            if len(self.get_bonded_atoms_to_i(atom_i)) in pi_valencies[self.get_atom_label(atom_i)]:
                return True
        return False

    def set_pi_bonds(self):
        pi_bonds = []
        if self.graph is None:
            logger.error('Could not find pi bonds')
            return

        bonds = list(self.graph.edges)
        for bond in bonds:
            if self.is_possible_pi_atom(bond[0]) and self.is_possible_pi_atom(bond[1]):
                pi_bonds.append(bond)

        if len(pi_bonds) > 0:
            self.pi_bonds = pi_bonds

    def _init_smiles(self, name, smiles):

        try:
            self.mol_obj = Chem.MolFromSmiles(smiles)
            self.mol_obj = Chem.AddHs(self.mol_obj)
        except RuntimeError:
            logger.critical(f'Could not generate an rdkit mol object for {name}')
            exit()

        self.n_atoms = self.mol_obj.GetNumAtoms()
        rdkit_bonds = self.mol_obj.GetBonds()
        self.n_bonds = len(rdkit_bonds)
        self.charge = Chem.GetFormalCharge(self.mol_obj)
        n_radical_electrons = rdkit.Chem.Descriptors.NumRadicalElectrons(self.mol_obj)
        self.mult = self._calc_multiplicity(n_radical_electrons)

        if self.n_atoms == 1:
            self.charges = [self.charge]

        AllChem.EmbedMultipleConfs(self.mol_obj, numConfs=1, params=AllChem.ETKDG())
        self.xyzs = extract_xyzs_from_rdkit_mol_object(self, conf_ids=[0])[0]

        unassigned_stereocentres = False
        stereocentres = []
        stereochem_info = Chem.FindMolChiralCenters(self.mol_obj, includeUnassigned=True)
        for atom, assignment in stereochem_info:
            if assignment == '?':
                unassigned_stereocentres = True
            else:
                stereocentres.append(atom)

        pi_bonds = []
        for bond in rdkit_bonds:
            if bond.GetBondType() != Chem.rdchem.BondType.SINGLE:
                pi_bonds.append((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()))
            if bond.GetStereo() != Chem.rdchem.BondStereo.STEREONONE:
                stereocentres.append(bond.GetBeginAtomIdx())
                stereocentres.append(bond.GetEndAtomIdx())

        if len(pi_bonds) > 0:
            self.pi_bonds = pi_bonds

        if len(stereocentres) > 0:
            self.stereocentres = stereocentres
            if unassigned_stereocentres:
                logger.warning('Unassigned stereocentres found')

        bond_list = get_bond_list_from_rdkit_bonds(rdkit_bonds_obj=self.mol_obj.GetBonds())

        if not rdkit_conformer_geometries_are_resonable(conf_xyzs=[self.xyzs]):
            logger.info('RDKit conformer was not reasonable')
            self.rdkit_conf_gen_is_fine = False
            self.xyzs = gen_simanl_conf_xyzs(name=self.name, init_xyzs=self.xyzs, bond_list=bond_list,
                                             stereocentres=self.stereocentres, n_simanls=1)[0]
            self.graph = mol_graphs.make_graph(self.xyzs, self.n_atoms)
            self.n_bonds = self.graph.number_of_edges()

        else:
            self.graph = mol_graphs.make_graph(self.xyzs, self.n_atoms, bond_list)
            self._check_rdkit_graph_agreement()

        self.distance_matrix = calc_distance_matrix(self.xyzs)

    def _init_xyzs(self, xyzs):
        for xyz in xyzs:
            if len(xyz) != 4:
                logger.critical(f'XYZ input is not the correct format (needs to be e.g. [\'H\',0,0,0]). '
                                f'Found {xyz} instead')
                exit()

        if isinstance(self, (Reactant, Product)):
            logger.warning('Initiating a molecule from xyzs means any stereocentres will probably be lost. Initiate '
                           'from a SMILES string to keep stereochemistry')

        self.n_atoms = len(xyzs)
        self.n_bonds = len(get_xyz_bond_list(xyzs))
        self.graph = mol_graphs.make_graph(self.xyzs, self.n_atoms)
        self.distance_matrix = calc_distance_matrix(xyzs)
        if self.n_atoms == 1:
            self.charges = [self.charge]
        self.set_pi_bonds()

    def __init__(self, name='molecule', smiles=None, xyzs=None, solvent=None, charge=0, mult=1, is_fragment=False):
        """Initialise a Molecule object.
        Will generate xyz lists of all the conformers found by RDKit within the number
        of conformers searched (n_confs)

        Keyword Arguments:
            name (str): Name of the molecule (default: {'molecule'})
            smiles (str): Standard SMILES string. e.g. generated by Chemdraw (default: {None})
            xyzs (list(list)): e.g. [['C', 0.0, 0.0, 0.0], ...] (default: {None})
            solvent (str): Solvent that the molecule is immersed in. Will be used in optimise() and single_point() (default: {None})
            charge (int): charge on the molecule (default: {0})
            mult (int): spin multiplicity on the molecule (default: {1})
            is_fragment (bool): if the molecule is a fragment of a stripped molecule (default: {False})
        """
        logger.info(f'Generating a Molecule object for {name}')

        self.name = name
        self.smiles = smiles
        self.set_solvent(solvent)
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
        self.charges = None

        self.qm_solvent_xyzs = None
        self.mm_solvent_xyzs = None

        if smiles:
            self._init_smiles(name, smiles)

        if xyzs:
            self._init_xyzs(xyzs)


class Reactant(Molecule):
    pass


class Product(Molecule):
    pass
