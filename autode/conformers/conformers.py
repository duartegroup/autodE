import numpy as np

from typing import Optional, Union, TYPE_CHECKING
from rdkit import Chem

from autode.values import Distance, Energy
from autode.atoms import Atom, Atoms
from autode.config import Config
from autode.mol_graphs import make_graph, is_isomorphic
from autode.geom import calc_heavy_atom_rmsd
from autode.log import logger
from autode.utils import ProcessPool
from autode.exceptions import NoConformers


if TYPE_CHECKING:
    from autode.conformers.conformer import Conformer
    from autode.wrappers.methods import Method
    from autode.mol_graphs import MolecularGraph
    from autode.wrappers.keywords import Keywords


def _calc_conformer(conformer, calc_type, method, keywords, n_cores=1):
    """Top-level hashable function to call in parallel"""
    getattr(conformer, calc_type)(
        method=method, keywords=keywords, n_cores=n_cores
    )
    return conformer


class Conformers(list):
    @property
    def lowest_energy(self) -> Optional["Conformer"]:
        """
        Return the lowest energy conformer state from this set. If no
        conformers have an energy then return None

        -----------------------------------------------------------------------
        Returns:
            (autode.conformers.Conformer | None): Conformer
        """
        if all(c.energy is None for c in self):
            logger.error("Have no conformers with an energy, so no lowest")
            return None

        energies = [c.energy if c.energy is not None else np.inf for c in self]
        return self[np.argmin(energies)]

    def prune(
        self,
        e_tol: Union[Energy, float] = Energy(1.0, "kJ mol-1"),
        rmsd_tol: Union[Distance, float, None] = None,
        n_sigma: float = 5,
        remove_no_energy: bool = False,
    ) -> None:
        """
        Prune conformers based on both energy and root mean squared deviation
        (RMSD) values. Will discard any conformers that are within e_tol in
        energy (Ha) and rmsd in RMSD (Å) to any other

        -----------------------------------------------------------------------
        Arguments:
            e_tol (Energy): Energy tolerance

            rmsd_tol (Distance | None): RMSD tolerance. Defaults to
                                        autode.Config.rmsd_threshold

            n_sigma (float | int):

            remove_no_energy (bool):
        """

        if remove_no_energy:
            self.remove_no_energy()

        self.prune_on_energy(e_tol=e_tol, n_sigma=n_sigma)
        self.prune_on_rmsd(rmsd_tol=rmsd_tol)

        return None

    def prune_on_energy(
        self,
        e_tol: Union[Energy, float] = Energy(1.0, "kJ mol-1"),
        n_sigma: float = 5,
    ) -> None:
        """
        Prune the conformers based on an energy threshold, discarding those
        that have energies that are similar to within e_tol. Also discards
        conformers with very high energies (indicating a problem
        with the calculation) if the are more than n_sigma standard deviations
        away from the mean

        -----------------------------------------------------------------------
        Arguments:
            e_tol (autode.values.Energy | float | None):

            n_sigma (int): Number of standard deviations a conformer energy
                   must be from the average for it not to be added
        """
        idxs_with_energy = [
            idx for idx, conf in enumerate(self) if conf.energy is not None
        ]
        n_prev_confs = len(self)

        if len(idxs_with_energy) < 2:
            logger.info(
                f"Only have {len(self)} conformers with an energy. No "
                f"need to prune"
            )
            return None

        energies = [self[idx].energy for idx in idxs_with_energy]

        # Use a lower-bounded σ to prevent division by zero
        std_dev_e = max(float(np.std(energies)), 1e-8)
        avg_e = np.average(energies)

        logger.info(
            f"Have {len(energies)} energies with μ={avg_e:.6f} Ha "
            f"σ={std_dev_e:.6f} Ha"
        )

        if isinstance(e_tol, Energy):
            e_tol = float(e_tol.to("Ha"))
        else:
            logger.warning(
                f"Assuming energy tolerance {e_tol:.6f} has units " f"of Ha"
            )

        # Delete from the end of the list to preserve the order when deleting
        for i, idx in enumerate(reversed(idxs_with_energy)):
            conf = self[idx]
            idxs_with_energy = [j for j in idxs_with_energy if j < len(self)]

            if np.abs(conf.energy - avg_e) / std_dev_e > n_sigma:
                logger.warning(
                    f"Conformer {idx} had an energy >{n_sigma}σ "
                    f"from the average - removing"
                )
                del self[idx]
                continue

            if i == 0:
                # The first (last) conformer must be unique
                continue

            if any(
                np.abs(conf.energy - self[o_idx].energy) < e_tol
                for o_idx in idxs_with_energy
                if o_idx != idx
            ):
                logger.info(f"Conformer {idx} had a non unique energy")
                del self[idx]
                continue

        logger.info(
            f"Stripped {n_prev_confs - len(self)} conformer(s)."
            f" {n_prev_confs} -> {len(self)}"
        )
        return None

    def prune_on_rmsd(
        self, rmsd_tol: Union[Distance, float, None] = None
    ) -> None:
        """
        Given a list of conformers add those that are unique based on an RMSD
        tolerance. If rmsd=None then use autode.Config.rmsd_threshold

        -----------------------------------------------------------------------
        Arguments:
            rmsd_tol (autode.values.Distance | float | None):
        """
        if len(self) < 2:
            logger.info(
                f"Only have {len(self)} conformers. No need to prune "
                f"on RMSD"
            )
            return None

        rmsd_tol = Config.rmsd_threshold if rmsd_tol is None else rmsd_tol

        if isinstance(rmsd_tol, float):
            logger.warning(
                f"Assuming RMSD tolerance {rmsd_tol:.2f} has units" f" of Å"
            )
            rmsd_tol = Distance(rmsd_tol, "Å")

        logger.info(
            f'Removing conformers with RMSD < {rmsd_tol.to("ang")} Å '
            f"to any other (heavy atoms only, with no symmetry)"
        )

        # Only enumerate up to but not including the final index, as at
        # least one of the conformers must be unique in geometry
        for idx in reversed(range(len(self) - 1)):
            conf = self[idx]

            if any(
                calc_heavy_atom_rmsd(conf.atoms, other.atoms) < rmsd_tol
                for o_idx, other in enumerate(self)
                if o_idx != idx
            ):
                logger.info(
                    f"Conformer {idx} was close in geometry to at "
                    f"least one other - removing"
                )

                del self[idx]

        logger.info(f"Pruned to {len(self)} unique conformer(s) on RMSD")
        return None

    def prune_diff_graph(self, graph: "MolecularGraph") -> None:
        """
        Remove conformers with a different molecular graph to a defined
        reference. Although all conformers should have the same molecular
        graph there are situations where not pruning these is useful

        -----------------------------------------------------------------------

        Arguments:
            graph: Reference graph
        """
        n_prev_confs = len(self)

        for idx in reversed(range(len(self))):
            conformer = self[idx]
            make_graph(conformer)

            if not is_isomorphic(
                conformer.graph, graph, ignore_active_bonds=True
            ):
                logger.warning("Conformer had a different graph. Ignoring")
                del self[idx]

        logger.info(f"Pruned on connectivity {n_prev_confs} -> {len(self)}")
        return None

    def remove_no_energy(self) -> None:
        """Remove all conformers from this list that do not have an energy"""
        n_conformers_before_remove = len(self)

        for idx in reversed(range(len(self))):  # Enumerate backwards
            if self[idx].energy is None:
                del self[idx]

        n_conformers = len(self)
        if n_conformers == 0 and n_conformers != n_conformers_before_remove:
            raise NoConformers(
                f"Removed all the conformers "
                f"{n_conformers_before_remove} -> 0"
            )

    def _parallel_calc(self, calc_type, method, keywords):
        """
        Run a set of calculations (single point energy evaluations or geometry
        optimisations) in parallel over every conformer in this set. Will
        attempt to use all autode.Config.n_cores as fully as possible

        Arguments:
            calc_type (str):

            method (autode.wrappers.base.ElectronicStructureMethod):

            keywords (autode.wrappers.keywords.Keywords):
        """
        # TODO: Test efficiency + improve with dynamic load balancing
        if len(self) == 0:
            logger.error(f"Cannot run {calc_type} over 0 conformers")
            return None

        n_cores_pp = max(Config.n_cores // len(self), 1)

        with ProcessPool(max_workers=Config.n_cores // n_cores_pp) as pool:
            jobs = [
                pool.submit(
                    _calc_conformer,
                    conf,
                    calc_type,
                    method,
                    keywords,
                    n_cores=n_cores_pp,
                )
                for conf in self
            ]

            for idx, res in enumerate(jobs):
                self[idx] = res.result()

        return None

    def optimise(
        self,
        method: "Method",
        keywords: Optional["Keywords"] = None,
    ) -> None:
        """
        Optimise a set of conformers in parallel

        -----------------------------------------------------------------------
        Arguments:
            method (autode.wrappers.base.ElectronicStructureMethod):

            keywords (autode.wrappers.keywords.Keywords):
        """
        return self._parallel_calc("optimise", method, keywords)

    def single_point(
        self,
        method: "Method",
        keywords: Optional["Keywords"] = None,
    ) -> None:
        """
        Evaluate single point energies for a set of conformers in parallel

        -----------------------------------------------------------------------
        Arguments:
            method (autode.wrappers.base.ElectronicStructureMethod):

            keywords (autode.wrappers.keywords.Keywords):
        """
        return self._parallel_calc("single_point", method, keywords)

    def copy(self) -> "Conformers":
        return Conformers([conformer.copy() for conformer in self])


def atoms_from_rdkit_mol(rdkit_mol_obj: Chem.Mol, conf_id: int = 0) -> Atoms:
    """
    Generate atoms for a conformer contained within an RDKit molecule object

    ---------------------------------------------------------------------------
    Arguments:
        rdkit_mol_obj (rdkit.Chem.Mol): RDKit molecule

        conf_id (int): Conformer id to convert to atoms

    Returns:
        (list(autode.atoms.Atom)): Atoms
    """

    mol_block_lines = Chem.MolToMolBlock(rdkit_mol_obj, confId=conf_id).split(
        "\n"
    )
    mol_file_atoms = Atoms()

    # Extract atoms from the mol block
    for line in mol_block_lines:
        split_line = line.split()

        if len(split_line) == 16:
            x, y, z, atom_label = split_line[:4]
            mol_file_atoms.append(Atom(atom_label, x=x, y=y, z=z))

    return mol_file_atoms
