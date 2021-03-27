import rdkit
from multiprocessing import Pool
from rdkit.Chem import AllChem
from autode.log.methods import methods
from autode.input_output import xyz_file_to_atoms
from autode.conformers.conformer import get_conformer
from autode.conformers.conf_gen import get_simanl_atoms
from autode.conformers.conformers import conf_is_unique_rmsd
from autode.conformers.conformers import atoms_from_rdkit_mol
from autode.atoms import metals
from autode.config import Config
from autode.log import logger
from autode.mol_graphs import make_graph
from autode.smiles.smiles import init_organic_smiles
from autode.smiles.smiles import init_smiles
from autode.species.species import Species
from autode.utils import requires_atoms


class Molecule(Species):

    def _init_smiles(self, smiles):
        """Initialise a molecule from a SMILES string using RDKit if it's
        purely organic"""

        if any(metal in smiles for metal in metals):
            init_smiles(self, smiles)

        else:
            init_organic_smiles(self, smiles)

        logger.info(f'Initialisation with SMILES successful. '
                    f'Charge={self.charge}, Multiplicity={self.mult}, '
                    f'Num. Atoms={self.n_atoms}')
        return None

    def _init_xyz_file(self, xyz_filename):
        """Initialise a molecule from a .xyz file"""
        logger.info('Generating species from .xyz file')

        self.atoms = xyz_file_to_atoms(xyz_filename)

        if (sum(atom.atomic_number for atom in self.atoms) % 2 != 0
            and self.charge % 2 == 0 and self.mult == 1):
            raise ValueError('Initialised a molecule from an xyz file with  '
                             'an odd number of electrons but had an even '
                             f'charge and S = {1}. Impossible!')

        # Override the default name with something more descriptive
        if self.name == 'molecule' or self.name.endswith('.xyz'):
            self.name = xyz_filename.rstrip('.xyz')

        make_graph(self)
        return None

    @requires_atoms()
    def _generate_conformers(self, n_confs=None):
        """
        Use a simulated annealing approach to generate conformers for this
        molecule.

        Keyword Arguments:
            n_confs (int): Number of conformers requested if None default to
            autode.Config.num_conformers
        """

        n_confs = n_confs if n_confs is not None else Config.num_conformers
        self.conformers = []

        if self.smiles is not None and self.rdkit_conf_gen_is_fine:
            logger.info(f'Using RDKit to gen conformers. {n_confs} requested')

            m_string = 'ETKDGv3' if hasattr(AllChem, 'ETKDGv3') else 'ETKDGv2'
            logger.info(f'Using the {m_string} method')

            method_class = getattr(AllChem, m_string)
            method = method_class()
            method.pruneRmsThresh = Config.rmsd_threshold
            method.numThreads = Config.n_cores
            method.useSmallRingTorsion = True

            logger.info('Running conformation generation with RDKit... running')
            conf_ids = list(AllChem.EmbedMultipleConfs(self.rdkit_mol_obj,
                                                       numConfs=n_confs,
                                                       params=method))
            logger.info('                                          ... done')

            conf_atoms_list = [atoms_from_rdkit_mol(self.rdkit_mol_obj, conf_id)
                               for conf_id in conf_ids]

            methods.add(f'{m_string} algorithm (10.1021/acs.jcim.5b00654) '
                        f'implemented in RDKit v. {rdkit.__version__}')

        else:
            logger.info('Using repulsion+relaxed (RR) to generate conformers')
            with Pool(processes=Config.n_cores) as pool:
                results = [pool.apply_async(get_simanl_atoms, (self, None, i))
                           for i in range(n_confs)]
                conf_atoms_list = [res.get(timeout=None) for res in results]

            methods.add('RR algorithm (???) implemented in autodE')

        # Add the unique conformers
        for i, atoms in enumerate(conf_atoms_list):
            conf = get_conformer(name=f'{self.name}_conf{i}', species=self)
            conf.atoms = atoms

            # If the conformer is unique on an RMSD threshold
            if conf_is_unique_rmsd(conf, self.conformers):
                self.conformers.append(conf)

        logger.info(f'Generated {len(self.conformers)} unique conformer(s)')
        return None

    def populate_conformers(self, n_confs):
        """Populate self.conformers with a list of Conformer objects"""
        return self._generate_conformers(n_confs=n_confs)

    def __init__(self, name='molecule', smiles=None, atoms=None,
                 solvent_name=None, charge=0, mult=1):
        """
        Molecule class

        Keyword Arguments:
            name (str): Name of the molecule or a .xyz filename

            smiles (str): Standard SMILES string. e.g. generated by
                          Chemdraw

            atoms (list(autode.atoms.Atom)): List of atoms in the species

            solvent_name (str): Solvent that the molecule is immersed in

            charge (int): Charge on the molecule

            mult (int): Spin multiplicity on the molecule
        """
        logger.info(f'Generating a Molecule object for {name}')
        super().__init__(name, atoms, charge, mult, solvent_name)

        if name.endswith('.xyz'):
            self._init_xyz_file(xyz_filename=name)

        self.smiles = smiles
        self.rdkit_mol_obj = None
        self.rdkit_conf_gen_is_fine = True

        self.conformers = None

        if smiles is not None:
            self._init_smiles(smiles)

        elif atoms is not None:
            make_graph(self)

        # If the name is unassigned use a more interpretable chemical formula
        if name == 'molecule' and self.atoms is not None:
            self.name = self.formula()


class SolvatedMolecule(Molecule):

    @requires_atoms()
    def optimise(self, method):
        raise NotImplementedError

    def __init__(self, name='solvated_molecule', smiles=None, atoms=None,
                 solvent_name=None, charge=0, mult=1, solvent_mol=None):
        super().__init__(name, smiles, atoms, solvent_name, charge, mult)

        self.solvent_mol = solvent_mol
        self.qm_solvent_atoms = None
        self.mm_solvent_atoms = None


class Reactant(Molecule):
    pass


class Product(Molecule):
    pass


def reactant_to_product(reactant):
    reactant.__class__ = Product
    return reactant


def product_to_reactant(product):
    product.__class__ = Reactant
    return product
