import re
import rdkit
from pathlib import Path
from typing import Optional, List, Any
from rdkit.Chem import AllChem

from autode.log.methods import methods
from autode.input_output import xyz_file_to_atoms, attrs_from_xyz_title_line
from autode.conformers.conformer import Conformer
from autode.conformers.conf_gen import get_simanl_conformer
from autode.conformers.conformers import atoms_from_rdkit_mol
from autode.atoms import metals, Atom
from autode.config import Config
from autode.log import logger
from autode.mol_graphs import make_graph
from autode.smiles.smiles import init_organic_smiles
from autode.smiles.smiles import init_smiles
from autode.species.species import Species
from autode.utils import requires_atoms, ProcessPool


class Molecule(Species):
    def __init__(
        self,
        arg: str = "molecule",
        smiles: Optional[str] = None,
        atoms: Optional[List["Atom"]] = None,
        solvent_name: Optional[str] = None,
        charge: Optional[int] = None,
        mult: Optional[int] = None,
        **kwargs,
    ):
        """
        A molecular species constructable from SMILES or a set of atoms,
        has default charge and spin multiplicity.

        -----------------------------------------------------------------------
        Arguments:
            arg: Name of the molecule or a .xyz filename

            smiles: Standard SMILES string. e.g. generated by Chemdraw

            atoms: List of atoms in the species

            solvent_name: Solvent that the molecule is immersed in

            charge: Charge on the molecule. If unspecified defaults to, if
                    present, that defined in a xyz file or 0

            mult: Spin multiplicity on the molecule. If unspecified defaults to, if
                  present, that defined in a xyz file or 1

        Keyword Arguments:
            name: Name of the molecule. Overrides arg if arg is not a .xyz
                  filename
        """
        logger.info(f"Generating a Molecule object for {arg}")
        super().__init__(
            name=arg if "name" not in kwargs else kwargs["name"],
            atoms=atoms,
            charge=charge if charge is not None else 0,
            mult=mult if mult is not None else 1,
            solvent_name=solvent_name,
        )
        self.smiles = smiles
        self.rdkit_mol_obj = None
        self.rdkit_conf_gen_is_fine = True

        if _is_xyz_filename(arg):
            self._init_xyz_file(
                xyz_filename=arg,
                charge=charge,
                mult=mult,
                solvent_name=solvent_name,
            )

        if smiles is not None:
            assert not _is_xyz_filename(arg), "Can't be both SMILES and file"
            self._init_smiles(smiles, charge=charge)

        # If the name is unassigned use a more interpretable chemical formula
        if self.name == "molecule" and self.atoms is not None:
            self.name = self.formula

    def __repr__(self):
        return self._repr(prefix="Molecule")

    def __eq__(self, other):
        """Equality of two molecules is only dependent on the identity"""
        return super().__eq__(other)

    def _init_smiles(self, smiles: str, charge: Optional[int]):
        """Initialise a molecule from a SMILES string using RDKit if it's
        purely organic.

        -----------------------------------------------------------------------
        Arguments:
            smiles (str):
        """
        at_strings = re.findall(r"\[.*?]", smiles)

        if any(metal in string for metal in metals for string in at_strings):
            init_smiles(self, smiles)

        else:
            init_organic_smiles(self, smiles)

        if charge is not None and charge != self._charge:
            raise ValueError(
                "SMILES charge was not the same as the "
                f"defined value. {self._charge} ≠ {charge}"
            )

        logger.info(
            f"Initialisation with SMILES successful. "
            f"Charge={self.charge}, Multiplicity={self.mult}, "
            f"Num. Atoms={self.n_atoms}"
        )
        return None

    def _init_xyz_file(self, xyz_filename: str, **override_attrs: Any):
        """
        Initialise a molecule from a .xyz file

        -----------------------------------------------------------------------
        Arguments:
            xyz_filename (str):
            charge (int | None):
            mult (int | None):

        Raises:
            (ValueError)
        """
        logger.info("Generating a species from .xyz file")

        self.atoms = xyz_file_to_atoms(xyz_filename)
        title_line_attrs = attrs_from_xyz_title_line(xyz_filename)
        logger.info(f"Found {title_line_attrs} in title line")

        for attr in ("charge", "mult", "solvent_name"):
            if override_attrs[attr] is None and attr in title_line_attrs:
                setattr(self, attr, title_line_attrs[attr])

        if (
            sum(atom.atomic_number for atom in self.atoms) % 2 != 0
            and self.charge % 2 == 0
            and self.mult == 1
        ):
            raise ValueError(
                "Initialised a molecule from an xyz file with  "
                "an odd number of electrons but had an even "
                "charge and 2S + 1 = 1. Impossible!"
            )

        # Override the default name with something more descriptive
        if self.name == "molecule" or _is_xyz_filename(self.name):
            self.name = Path(self.name).stem

        make_graph(self)
        return None

    @requires_atoms
    def _generate_conformers(self, n_confs: Optional[int] = None):
        """
        Use a simulated annealing approach to generate conformers for this
        molecule.

        -----------------------------------------------------------------------
        Keyword Arguments:
            n_confs (int): Number of conformers requested if None default to
                           autode.Config.num_conformers
        """

        n_confs = n_confs if n_confs is not None else Config.num_conformers
        self.conformers.clear()

        if self.smiles is not None and self.rdkit_conf_gen_is_fine:
            logger.info(f"Using RDKit to gen conformers. {n_confs} requested")

            m_string = "ETKDGv3" if hasattr(AllChem, "ETKDGv3") else "ETKDGv2"
            logger.info(f"Using the {m_string} method")

            method_class = getattr(AllChem, m_string)
            method = method_class()
            method.pruneRmsThresh = Config.rmsd_threshold
            method.numThreads = Config.n_cores
            method.useSmallRingTorsion = True

            logger.info(
                "Running conformation generation with RDKit... running"
            )
            conf_ids = list(
                AllChem.EmbedMultipleConfs(
                    self.rdkit_mol_obj, numConfs=n_confs, params=method
                )
            )
            logger.info("                                          ... done")

            for conf_id in conf_ids:
                conf = Conformer(
                    species=self, name=f"{self.name}_conf{conf_id}"
                )
                conf.atoms = atoms_from_rdkit_mol(self.rdkit_mol_obj, conf_id)
                self.conformers.append(conf)

            methods.add(
                f"{m_string} algorithm (10.1021/acs.jcim.5b00654) "
                f"implemented in RDKit v. {rdkit.__version__}"
            )

        else:
            logger.info("Using repulsion+relaxed (RR) to generate conformers")
            with ProcessPool(max_workers=Config.n_cores) as pool:
                results = [
                    pool.submit(get_simanl_conformer, self, None, i)
                    for i in range(n_confs)
                ]
                self.conformers = [res.result() for res in results]  # type: ignore

            self.conformers.prune_on_energy(e_tol=1e-10)
            methods.add(
                "RR algorithm (10.1002/anie.202011941) implemented in autodE"
            )

        self.conformers.prune_on_rmsd()
        return None

    def populate_conformers(self, n_confs: Optional[int] = None) -> None:
        """
        Populate self.conformers with a conformers generated using a default
        method

        -----------------------------------------------------------------------
        Arguments:
            n_confs (int): Number of conformers to try and generate
        """
        return self._generate_conformers(n_confs=n_confs)

    def to_product(self) -> "Product":
        """
        Generate a copy of this reactant as a product

        -----------------------------------------------------------------------
        Returns:
            (autode.species.molecule.Product): Product
        """
        product = self.copy()  # type: ignore
        product.__class__ = Product

        return product  # type: ignore

    def to_reactant(self) -> "Reactant":
        """
        Generate a copy of this product as a reactant

        -----------------------------------------------------------------------
        Returns:
            (autode.species.molecule.Reactant): Reactant
        """
        reactant = self.copy()  # type: ignore
        reactant.__class__ = Reactant

        return reactant  # type: ignore


class Reactant(Molecule):
    """Reactant molecule"""


class Product(Molecule):
    """Product molecule"""


def _is_xyz_filename(value: str) -> bool:
    return isinstance(value, str) and value.endswith(".xyz")
