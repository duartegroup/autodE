from autode.config import Config
from autode.log import logger
from rdkit.Chem import AllChem
from rdkit import Chem
import rdkit.Chem.Descriptors
from autode.species import Species
from autode.geom import are_coords_reasonable
from autode.mol_graphs import make_graph
from autode.conformers.conformers import get_atoms_from_rdkit_mol_object
from autode.conformers.conformer import Conformer
from autode.conformers.conf_gen import get_simanl_atoms
from autode.calculation import Calculation
from autode.solvent.explicit_solvent import do_explicit_solvent_qmmm
from autode.exceptions import RDKitFailed, BondsInSMILESAndGraphDontMatch, NoAtomsInMolecule
from autode.utils import requires_atoms


class Molecule(Species):

    def _init_smiles(self, smiles):
        """Initialise a molecule from a SMILES string using RDKit"""

        try:
            self.rdkit_mol_obj = Chem.MolFromSmiles(smiles)
            self.rdkit_mol_obj = Chem.AddHs(self.rdkit_mol_obj)
        except RuntimeError:
            raise RDKitFailed

        self.charge = Chem.GetFormalCharge(self.rdkit_mol_obj)
        self.mult = self._calc_multiplicity(rdkit.Chem.Descriptors.NumRadicalElectrons(self.rdkit_mol_obj))

        # Generate a single 3D structure using RDKit's ETKDG conformer generation algorithm
        AllChem.EmbedMultipleConfs(self.rdkit_mol_obj, numConfs=1, params=AllChem.ETKDGv2())
        self.set_atoms(atoms=get_atoms_from_rdkit_mol_object(self.rdkit_mol_obj, conf_id=0))

        if not are_coords_reasonable(coords=self.get_coordinates()):
            logger.warning('RDKit conformer was not reasonable')
            self.rdkit_conf_gen_is_fine = False

            make_graph(self, rdkit_bonds=self.rdkit_mol_obj.GetBonds())
            self.set_atoms(atoms=get_simanl_atoms(self))

        # Ensure the SMILES string and the 3D structure have the same bonds
        make_graph(self)

        if len(self.rdkit_mol_obj.GetBonds()) != self.graph.number_of_edges():
            raise BondsInSMILESAndGraphDontMatch

        logger.info(f'Initialisation with SMILES successful. Charge={self.charge}, Multiplicity={self.mult}, '
                    f'Num. Atoms={self.n_atoms}')
        return None

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

    @requires_atoms()
    def _find_stereocentres(self):
        # TODO... write this

        # Set self.graph.node['stereo'] = True
        pass

    @requires_atoms()
    def _generate_conformers(self, n_rdkit_confs=300, n_siman_confs=50):
        """
        Use either RDKit or a simulated annealing approach to generate conformers for this molecule. RDKit is preferred,
        being considerably faster. However for some unusual molecule and metal complexes it fails to generate a sensible
        structure. In this case fall back to the simulated annealing algorithm

        Keyword Arguments:
            n_rdkit_confs (int):
            n_siman_confs (int):
        """
        self.conformers = []

        if self.smiles is not None and self.rdkit_conf_gen_is_fine:
            logger.info(f'Using RDKit to generate conformers. {n_rdkit_confs} requested')

            method = AllChem.ETKDGv2()
            method.pruneRmsThresh = 0.5
            method.numThreads = Config.n_cores

            logger.info('Running conformation generation with RDKit... running')
            conf_ids = list(AllChem.EmbedMultipleConfs(self.rdkit_mol_obj, numConfs=n_rdkit_confs, params=method))
            logger.info('                                          ... done')

            conf_atoms_list = [get_atoms_from_rdkit_mol_object(self.rdkit_mol_obj, conf_id) for conf_id in conf_ids]

        else:
            logger.info('Using simulated annealing to generate conformers')
            conf_atoms_list = [get_simanl_atoms(species=self, conf_n=i) for i in range(n_siman_confs)]

        for i, atoms in enumerate(conf_atoms_list):
            conf = Conformer(name=f'{self.name}_conf{i}', atoms=atoms,  charge=self.charge, mult=self.mult)
            conf.solvent = self.solvent
            self.conformers.append(conf)

        logger.info(f'Generated {len(self.conformers)} unique conformer(s)')
        return None

    @requires_atoms()
    def optimise(self, method):
        logger.info(f'Running optimisation of {self.name}')

        opt = Calculation(name=f'{self.name}_opt', molecule=self, method=method,
                          keywords_list=method.keywords.opt, n_cores=Config.n_cores, opt=True)
        opt.run()
        self.energy = opt.get_energy()
        self.set_atoms(atoms=opt.get_final_atoms())
        self.print_xyz_file(filename=f'{self.name}_optimised_{method.name}.xyz')

        return None

    def __init__(self, name='molecule', smiles=None, atoms=None, solvent_name=None, charge=0, mult=1):
        """Initialise a Molecule object.
        Will generate xyz lists of all the conformers found by RDKit within the number
        of conformers searched (n_confs)

        Keyword Arguments:
            name (str): Name of the molecule (default: {'molecule'})
            smiles (str): Standard SMILES string. e.g. generated by Chemdraw (default: {None})
            atoms (list(autode.atoms.Atom)): List of atoms in the species (default: {None})
            solvent_name (str): Solvent that the molecule is immersed in (default: {None})
            charge (int): Charge on the molecule (default: {0})
            mult (int): Spin multiplicity on the molecule (default: {1})
        """
        logger.info(f'Generating a Molecule object for {name}')
        super().__init__(name, atoms, charge, mult, solvent_name)

        self.smiles = smiles
        self.rdkit_conf_gen_is_fine = True
        self.rdkit_mol_obj = None

        self.conformers = None

        if smiles:
            self._init_smiles(smiles)

        if self.n_atoms == 0:
            raise NoAtomsInMolecule

        self._find_stereocentres()
        make_graph(self)


class SolvatedMolecule(Molecule):

    def optimise(self, method):
        logger.info(f'Running optimisation of {self.name}')

        opt = Calculation(name=self.name + '_opt', molecule=self, method=method, keywords_list=method.opt_keywords,
                          n_cores=Config.n_cores, opt=True)
        opt.run()
        self.energy = opt.get_energy()
        self.set_atoms(atoms=opt.get_final_atoms())
        self.charges = opt.get_atomic_charges()

        # TODO get this to return the atoms
        _, qmmm_xyzs, n_qm_atoms = do_explicit_solvent_qmmm(self, self.solvent, method, n_confs=96, n_qm_solvent_mols=30)
        self.xyzs = qmmm_xyzs[:self.n_atoms]
        self.qm_solvent_xyzs = qmmm_xyzs[self.n_atoms: n_qm_atoms]
        self.mm_solvent_xyzs = qmmm_xyzs[n_qm_atoms:]

    def __init__(self, name='solvated_molecule', smiles=None, atoms=None, solvent_name=None, charge=0, mult=1):
        super().__init__(name, smiles, atoms, solvent_name, charge, mult)

        self.qm_solvent_xyzs = None
        self.mm_solvent_xyzs = None


class Reactant(Molecule):
    pass


class Product(Molecule):
    pass
