import rdkit.Chem.Descriptors
from rdkit.Chem import AllChem
from rdkit import Chem
from autode.conformers.conf_gen import get_simanl_atoms
from autode.conformers.conformers import atoms_from_rdkit_mol
from autode.exceptions import RDKitFailed
from autode.geom import are_coords_reasonable
from autode.log import logger
from autode.mol_graphs import make_graph
from autode.smiles.smiles_parser import parse_smiles
from copy import deepcopy


def calc_multiplicity(molecule, n_radical_electrons):
    """Calculate the spin multiplicity 2S + 1 where S is the number of
    unpaired electrons

    Arguments:
        molecule (autode.molecule.Molecule):
        n_radical_electrons (int):
    Returns:
        int: multiplicity of the molecule
    """

    if molecule.mult == 1 and n_radical_electrons == 0:
        return 1

    if molecule.mult == 1 and n_radical_electrons == 1:
        # Cannot have multiplicity = 1 and 1 radical electrons – override
        # default multiplicity
        return 2

    if molecule.mult == 1 and n_radical_electrons > 1:
        logger.warning('Diradicals by default singlets. Set mol.mult if it\'s '
                       'any different')
        return 1

    return molecule.mult


def init_organic_smiles(molecule, smiles):
    """
    Initialise a molecule from a SMILES string, set the charge, multiplicity (
    if it's not already specified) and the 3D geometry using RDKit

    Arguments:
        molecule (autode.molecule.Molecule):
        smiles (str): SMILES string
    """

    try:
        molecule.rdkit_mol_obj = Chem.MolFromSmiles(smiles)

        if molecule.rdkit_mol_obj is None:
            logger.warning('RDKit failed to initialise a molecule')
            return init_smiles(molecule, smiles)

        molecule.rdkit_mol_obj = Chem.AddHs(molecule.rdkit_mol_obj)
    except RuntimeError:
        raise RDKitFailed

    molecule.charge = Chem.GetFormalCharge(molecule.rdkit_mol_obj)
    molecule.mult = calc_multiplicity(molecule, rdkit.Chem.Descriptors.NumRadicalElectrons(molecule.rdkit_mol_obj))
    bonds = [(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()) for bond in molecule.rdkit_mol_obj.GetBonds()]

    # Generate a single 3D structure using RDKit's ETKDG conformer generation
    # algorithm
    method = AllChem.ETKDGv2()
    method.randomSeed = 0xf00d
    AllChem.EmbedMultipleConfs(molecule.rdkit_mol_obj, numConfs=1, params=method)
    molecule.atoms = atoms_from_rdkit_mol(molecule.rdkit_mol_obj, conf_id=0)
    make_graph(molecule, bond_list=bonds)

    # Revert back to RR if RDKit fails to return a sensible geometry
    if not are_coords_reasonable(coords=molecule.get_coordinates()):
        molecule.rdkit_conf_gen_is_fine = False
        molecule.atoms = get_simanl_atoms(molecule, save_xyz=False)

    for atom, _ in Chem.FindMolChiralCenters(molecule.rdkit_mol_obj):
        molecule.graph.nodes[atom]['stereo'] = True

    for bond in molecule.rdkit_mol_obj.GetBonds():
        if bond.GetBondType() != Chem.rdchem.BondType.SINGLE:
            molecule.graph.edges[bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()]['pi'] = True
        if bond.GetStereo() != Chem.rdchem.BondStereo.STEREONONE:
            molecule.graph.nodes[bond.GetBeginAtomIdx()]['stereo'] = True
            molecule.graph.nodes[bond.GetEndAtomIdx()]['stereo'] = True

    check_bonds(molecule, bonds=molecule.rdkit_mol_obj.GetBonds())

    return None


def init_smiles(molecule, smiles):
    """
    Initialise a molecule from a SMILES string
    Arguments:
        molecule (autode.molecule.Molecule):
        smiles (str): SMILES string
    """
    # Assume that the RDKit conformer generation algorithm is not okay for
    # metals
    molecule.rdkit_conf_gen_is_fine = False

    parser = parse_smiles(smiles)

    molecule.charge = parser.charge
    molecule.mult = calc_multiplicity(molecule=molecule,
                                      n_radical_electrons=parser.n_radical_electrons)

    molecule.atoms = parser.atoms

    make_graph(molecule, bond_list=parser.bonds)

    for stereocentre in parser.stereocentres:
        molecule.graph.nodes[stereocentre]['stereo'] = True
    for bond_index in parser.bond_order_dict.keys():
        bond = parser.bonds[bond_index]
        molecule.graph.edges[bond]['pi'] = True

    molecule.atoms = get_simanl_atoms(molecule, save_xyz=False)
    check_bonds(molecule, bonds=parser.bonds)

    return None


def check_bonds(molecule, bonds):
    """
    Ensure the SMILES string and the 3D structure have the same bonds,
    but don't override

    Arguments:
        molecule (autode.molecule.Molecule):
        bonds (list):
    """
    check_molecule = deepcopy(molecule)
    make_graph(check_molecule)

    if len(bonds) != check_molecule.graph.number_of_edges():
        logger.error('Bonds and graph do no match')

    return None
