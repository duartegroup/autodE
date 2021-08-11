from typing import Optional
from autode.values import Distance, Energy
from rdkit import Chem
from autode.atoms import Atom
from autode.config import Config
from autode.geom import calc_heavy_atom_rmsd
from autode.log import logger
import numpy as np


class Conformers(list):

    def prune(self,
              e_tol:    Energy = Energy(1, units='kJ mol-1'),
              rmsd_tol: Optional[Distance] = None,
              n_sigma:  float = 5) -> None:
        """
        Prune conformers based on both energy and root mean squared deviation
        (RMSD) values. Will discard any conformers that are within e_tol in
        energy (Ha) and rmsd in RMSD (Å) to any other

        Keyword Arguments:
            e_tol (Energy): Energy tolerance

            rmsd_tol  (Distance | None): RMSD tolerane

            n_sigma (float | int):
        """
        self.prune_on_energy(e_tol=e_tol, n_sigma=n_sigma)
        self.prune_on_rmsd(rmsd_tol=rmsd_tol)

        return None

    def prune_on_energy(self,
                        e_tol:   Energy = Energy(1, units='kJ mol-1'),
                        n_sigma: float = 5) -> None:
        """
        Prune the conformers based on an energy threshold, discarding those
        that have energies that are similar to within e_tol. Also discards
        conformers with very high energies (indicating a problem
        with the calculation) if the are more than n_sigma standard deviations
        away from the mean

        Arguments:
            e_tol (autode.values.Energy | float | None):

            n_sigma (int): Number of standard deviations a conformer energy
                   must be from the average for it not to be added
        """
        confs_with_energy = {idx: conf for idx, conf in enumerate(self)
                             if conf.energy is not None}
        n_prev_confs = len(self)

        if len(confs_with_energy) < 2:
            logger.info(f'Only had {len(self)} conformers with an energy. No '
                        f'need to prune')
            return None

        energies = [conf.energy for conf in confs_with_energy.values()]
        std_dev_e, avg_e = float(np.std(energies)), np.average(energies)

        logger.warning(f'Have {len(energies)} energies with μ={avg_e:.6f} Ha '
                       f'σ={std_dev_e:.6f} Ha')

        if isinstance(e_tol, float):
            logger.warning('Assuming energy tolerance has units of Ha')
            e_tol = Energy(e_tol, units='Ha')

        # Delete from the end of the list to preserve the order when deleting
        for i, (idx, conf) in enumerate(reversed(confs_with_energy.items())):

            if (np.abs(conf.energy - avg_e)
                 / max(std_dev_e, 1E-8)) > n_sigma:

                logger.warning(f'Conformer {idx} had an energy >{n_sigma}σ '
                               f'from the average - removing')
                del self[idx]

            if i == 0:
                # The first (last) conformer must be unique
                continue

            if any(np.abs(conf.energy - other.energy) < e_tol.to('Ha')
                   for j, other in confs_with_energy.items() if j != idx):

                logger.info(f'Conformer {idx} had a non unique energy')
                del self[idx]
                continue

        logger.info(f'Stripped {n_prev_confs - len(self)} conformer(s).'
                    f' {n_prev_confs} -> {len(self)}')
        return None

    def prune_on_rmsd(self,
                      rmsd_tol: Optional[Distance] = None) -> None:
        """
        Given a list of conformers add those that are unique based on an RMSD
        tolerance. If rmsd=None then use autode.Config.rmsd_threshold

        Arguments:
            rmsd_tol (autode.values.Distance | None):
        """
        if len(self) < 2:
            logger.info(f'Only had {len(self)}. No need to prune on RMSD')
            return

        rmsd_tol = Config.rmsd_threshold if rmsd_tol is None else rmsd_tol
        logger.info(f'Removing conformers with RMSD < {rmsd_tol.to("ang")} Å '
                    f'to any other (heavy atoms only, with no symmetry)')

        # Only enumerate up to but not including the final index, as at
        # least one of the conformers must be unique in geometry
        for idx in reversed(range(len(self)-1)):

            conf = self[idx]

            if all(calc_heavy_atom_rmsd(conf.atoms, other.atoms) < rmsd_tol
                   for o_idx, other in enumerate(self) if o_idx != idx):

                logger.info(f'Conformer {idx} was close in geometry to at'
                            f'least one other - removing')

                del self[idx]

        logger.info(f'Pruned to {len(self)} unique conformer(s) on RMSD')
        return None






def atoms_from_rdkit_mol(rdkit_mol_obj, conf_id):
    """Generate atoms for conformers in rdkit_mol_obj

    Arguments:
        rdkit_mol_obj (rdkit.Chem.Mol): RDKit molecule
        conf_id (int): Conformer id to convert to atoms

    Returns:
        (list(autode.atoms.Atom)): Atoms
    """

    mol_block_lines = Chem.MolToMolBlock(rdkit_mol_obj,
                                         confId=conf_id).split('\n')
    mol_file_atoms = []

    # Extract atoms from the mol block
    for line in mol_block_lines:
        split_line = line.split()

        if len(split_line) == 16:
            atom_label = split_line[3]
            x, y, z = split_line[0], split_line[1], split_line[2]
            mol_file_atoms.append(Atom(atom_label, x=x, y=y, z=z))

    return mol_file_atoms
