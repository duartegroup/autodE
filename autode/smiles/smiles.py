from rdkit.Chem import AllChem
from rdkit.Chem.Descriptors import NumRadicalElectrons
from rdkit import Chem
from autode.conformers.conf_gen import get_simanl_atoms
from autode.conformers.conformers import atoms_from_rdkit_mol
from autode.exceptions import RDKitFailed, SMILESBuildFailed
from autode.log import logger
from autode.mol_graphs import make_graph
from autode.smiles import Parser, Builder


def calc_multiplicity(molecule, n_radical_electrons):
    """Calculate the spin multiplicity 2S + 1 where S is the number of
    unpaired electrons. Will only override non-default (unity multiplicity)

    ---------------------------------------------------------------------------
    Arguments:
        molecule (autode.molecule.Molecule):

    Keyword Arguments:
        n_radical_electrons (int | None):

    Returns:
        (int): multiplicity of the molecule
    """

    if molecule.mult == 1 and n_radical_electrons == 1:
        # Cannot have multiplicity = 1 and 1 radical electrons – override
        # default multiplicity
        return 2

    if molecule.mult == 1 and n_radical_electrons > 1:
        logger.warning(
            "Diradicals by default singlets. Set mol.mult if it's "
            "any different"
        )
        return 1

    return molecule.mult


def init_organic_smiles(molecule, smiles):
    """
    Initialise a molecule from a SMILES string, set the charge, multiplicity (
    if it's not already specified) and the 3D geometry using RDKit

    ---------------------------------------------------------------------------
    Arguments:
        molecule (autode.molecule.Molecule):

        smiles (str): SMILES string
    """
    parser, builder = Parser(), Builder()

    parser.parse(smiles)
    builder.set_atoms_bonds(atoms=parser.atoms, bonds=parser.bonds)

    # RDKit struggles with large rings or single atoms
    if builder.max_ring_n >= 8 or builder.n_atoms == 1:
        logger.info("Falling back to autodE builder")

        # TODO: currently the SMILES is parsed twice, which is not ideal
        return init_smiles(molecule=molecule, smiles=smiles)

    try:
        rdkit_mol = Chem.MolFromSmiles(smiles)

        if rdkit_mol is None:
            logger.warning("RDKit failed to initialise a molecule")
            return init_smiles(molecule, smiles)

        rdkit_mol = Chem.AddHs(rdkit_mol)

    except RuntimeError:
        raise RDKitFailed

    logger.info("Using RDKit to initialise")

    molecule.charge = Chem.GetFormalCharge(rdkit_mol)
    molecule.mult = calc_multiplicity(molecule, NumRadicalElectrons(rdkit_mol))

    bonds = [
        (bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
        for bond in rdkit_mol.GetBonds()
    ]

    # Generate a single 3D structure using RDKit's ETKDG conformer generation
    # algorithm
    method = AllChem.ETKDGv2()
    method.randomSeed = 0xF00D
    AllChem.EmbedMultipleConfs(rdkit_mol, numConfs=1, params=method)
    molecule.atoms = atoms_from_rdkit_mol(rdkit_mol, conf_id=0)

    # Revert to RR if RDKit fails to return a sensible geometry
    if not molecule.has_reasonable_coordinates:
        molecule.rdkit_conf_gen_is_fine = False
        molecule.atoms = get_simanl_atoms(molecule, save_xyz=False)

    for atom, _ in Chem.FindMolChiralCenters(rdkit_mol):
        molecule.graph.nodes[atom]["stereo"] = True

    for bond in rdkit_mol.GetBonds():
        idx_i, idx_j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()

        if bond.GetBondType() != Chem.rdchem.BondType.SINGLE:
            molecule.graph.edges[idx_i, idx_j]["pi"] = True

        if bond.GetStereo() != Chem.rdchem.BondStereo.STEREONONE:
            molecule.graph.nodes[idx_i]["stereo"] = True
            molecule.graph.nodes[idx_j]["stereo"] = True

    for atom, smiles_atom in zip(molecule.atoms, parser.atoms):
        atom.atom_class = smiles_atom.atom_class

    make_graph(species=molecule, bond_list=bonds)
    check_bonds(molecule, bonds=rdkit_mol.GetBonds())

    molecule.rdkit_mol_obj = rdkit_mol
    return None


def init_smiles(molecule, smiles):
    """
    Initialise a molecule from a SMILES string

    ---------------------------------------------------------------------------
    Arguments:
        molecule (autode.molecule.Molecule):

        smiles (str): SMILES string
    """
    molecule.rdkit_conf_gen_is_fine = False

    parser, builder = Parser(), Builder()

    parser.parse(smiles)
    molecule.charge = parser.charge

    # Only override the default multiplicity (1) with the parser-defined value
    if molecule.mult == 1:
        molecule.mult = parser.mult

    try:
        builder.build(atoms=parser.atoms, bonds=parser.bonds)
        molecule.atoms = builder.canonical_atoms

    except (SMILESBuildFailed, NotImplementedError):
        molecule.atoms = builder.canonical_atoms_at_origin

    for idx, atom in enumerate(builder.atoms):
        if atom.has_stereochem:
            molecule.graph.nodes[idx]["stereo"] = True

    make_graph(molecule, bond_list=parser.bonds)
    check_bonds(molecule, bonds=parser.bonds)

    for bond in parser.bonds:
        molecule.graph.edges[tuple(bond)]["pi"] = True

    if not molecule.has_reasonable_coordinates:
        logger.warning(
            "3D builder did not make a sensible geometry, "
            "Falling back to random minimisation."
        )
        molecule.atoms = get_simanl_atoms(molecule, save_xyz=False)

    return None


def check_bonds(molecule, bonds):
    """
    Ensure the SMILES string and the 3D structure have the same bonds,
    but don't override

    ---------------------------------------------------------------------------
    Arguments:
        molecule (autode.molecule.Molecule):

        bonds (list):
    """
    check_molecule = molecule.copy()
    make_graph(check_molecule)

    if len(bonds) != check_molecule.graph.number_of_edges():
        logger.warning("Bonds and graph do not match")

    return None
