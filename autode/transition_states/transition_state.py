from typing import Optional, List, TYPE_CHECKING

from autode.values import Frequency
from autode.transition_states.base import displaced_species_along_mode
from autode.transition_states.base import TSbase
from autode.transition_states.ts_guess import TSguess
from autode.transition_states.templates import TStemplate
from autode.conformers.conformers import Conformers
from autode.input_output import atoms_to_xyz_file
from autode.calculations import Calculation
from autode.config import Config
from autode.exceptions import CalculationException
from autode.geom import calc_heavy_atom_rmsd
from autode.log import logger
from autode.methods import get_hmethod
from autode.mol_graphs import get_truncated_active_mol_graph
from autode.utils import requires_atoms, requires_graph, ProcessPool


if TYPE_CHECKING:
    from autode.species.species import Species
    from autode.wrappers.keywords import Keywords
    from autode.wrappers.methods import Method
    from autode.bond_rearrangement import BondRearrangement


class TransitionState(TSbase):
    def __init__(
        self,
        ts_guess: TSbase,
        bond_rearr: Optional["BondRearrangement"] = None,
    ):
        """
        Transition State

        -----------------------------------------------------------------------
        Arguments:
            ts_guess (autode.transition_states.ts_guess.TSguess):

        Keyword Arguments:
            bond_rearr (autode.bond_rearrangement.BondRearrangement):
        """
        super().__init__(
            atoms=ts_guess.atoms,
            reactant=ts_guess.reactant,
            product=ts_guess.product,
            name=f"TS_{ts_guess.name}",
            charge=ts_guess.charge,
            bond_rearr=ts_guess.bond_rearrangement,
            mult=ts_guess.mult,
        )

        self.energy = ts_guess.energy
        self.gradient = ts_guess.gradient
        self.hessian = ts_guess.hessian

        if bond_rearr is not None:
            self.bond_rearrangement = bond_rearr

        self.solvent = ts_guess.solvent
        self._update_graph()

        self.warnings = ""  #: str for any warnings that may arise

    def __repr__(self):
        return self._repr(prefix="TransitionState")

    def __eq__(self, other):
        """Equality of this TS to another"""
        return super().__eq__(other)

    @requires_graph
    def _update_graph(self) -> None:
        """Update the molecular graph to include all the bonds that are being
        made/broken"""
        assert self.graph is not None, "Must have a MolecularGraph"

        if self.bond_rearrangement is None:
            logger.warning(
                "Bond rearrangement not set - molecular graph "
                "updating with no active bonds"
            )
        else:
            for bond in self.bond_rearrangement.all:
                self.graph.add_active_edge(*bond)

        logger.info(f"Molecular graph updated with active bonds")
        return None

    def _run_opt_ts_calc(self, method: "Method", name_ext: str) -> None:
        """Run an optts calculation and attempt to set the geometry, energy and
        normal modes"""
        assert method.keywords.opt_ts is not None, "Must have OptTS keywords"
        optts_calc = Calculation(
            name=f"{self.name}_{name_ext}",
            molecule=self,
            method=method,
            n_cores=Config.n_cores,
            keywords=method.keywords.opt_ts,
        )
        try:
            optts_calc.run()

            if not optts_calc.optimiser.converged:
                self._reoptimise(optts_calc, name_ext, method)

        except CalculationException:
            logger.error("Transition state optimisation calculation failed")

        return None

    def _reoptimise(
        self, calc: Calculation, name_ext: str, method: "Method"
    ) -> Calculation:
        """Rerun a calculation for more steps"""

        if calc.optimiser.last_energy_change.to("kcal mol-1") > 0.1:
            self.warnings += f"TS for {self.name} was not fully converged."
            logger.info("Optimisation did not converge")
            return calc

        logger.info("Optimisation nearly converged")
        if not self.could_have_correct_imag_mode:
            logger.warning("Lost imaginary mode")
            return calc

        logger.info(
            "Still have correct imaginary mode, trying "
            "more  optimisation steps"
        )

        assert method.keywords.opt_ts is not None, "Must have OptTS keywords"

        calc = Calculation(
            name=f"{self.name}_{name_ext}_reopt",
            molecule=self,
            method=method,
            n_cores=Config.n_cores,
            keywords=method.keywords.opt_ts,
        )
        calc.run()

        return calc

    def _generate_conformers(self, n_confs: Optional[int] = None) -> None:
        """Generate conformers at the TS"""
        from autode.conformers.conf_gen import get_simanl_conformer

        n_confs = Config.num_conformers if n_confs is None else n_confs
        distance_consts = self.active_bond_constraints
        self.conformers.clear()

        with ProcessPool(max_workers=Config.n_cores) as pool:
            results = [
                pool.submit(get_simanl_conformer, self, distance_consts, i)
                for i in range(n_confs)
            ]

            self.conformers = [res.result() for res in results]  # type: ignore

        self.conformers.prune(e_tol=1e-6)
        return None

    @property
    def vib_frequencies(self) -> Optional[List[Frequency]]:
        """
        Vibrational frequencies, which are all but the lowest 7 as the 6th
        is the 'translational' mode over the TS

        -----------------------------------------------------------------------
        Returns:
            (list(autode.value.Frequency) | None):
        """
        n = 7 if not self.is_linear() else 6
        return self.frequencies[n:] if self.frequencies is not None else None

    @requires_atoms
    def print_imag_vector(
        self, mode_number: int = 6, name: Optional[str] = None
    ) -> None:
        """Print a .xyz file with multiple structures visualising the largest
        magnitude imaginary mode

        -----------------------------------------------------------------------
        Keyword Arguments:
            mode_number (int): Number of the normal mode to visualise,
                               6 (default) is the lowest frequency vibration
                               i.e. largest magnitude imaginary, if present
            name (str):
        """
        name = self.name if name is None else name

        disp = -0.5
        for i in range(40):
            disp_ts = displaced_species_along_mode(
                self, mode_number=int(mode_number), disp_factor=disp
            )
            atoms_to_xyz_file(
                atoms=disp_ts.atoms, filename=f"{name}.xyz", append=True
            )

            # Add displacement so the final set of atoms are +0.5 Å displaced
            # along the mode, then displaced back again
            sign = 1 if i < 20 else -1
            disp += sign * 1.0 / 20.0

        return None

    @requires_atoms
    def optimise(
        self,
        name_ext: str = "optts",
        method: Optional["Method"] = None,
        reset_graph: bool = False,
        calc: Optional[Calculation] = None,
        keywords: Optional["Keywords"] = None,
    ):
        """Optimise this TS to a true TS"""
        logger.info(f"Optimising {self.name} to a transition state")

        self._run_opt_ts_calc(method=get_hmethod(), name_ext=name_ext)

        # A transition state is a first order saddle point i.e. has a single
        # imaginary frequency
        if not self.has_imaginary_frequencies:
            logger.error(
                "Transition state optimisation did not return any "
                "imaginary frequencies"
            )
            return

        assert self.imaginary_frequencies is not None
        if len(self.imaginary_frequencies) == 1:
            logger.info("Found a TS with a single imaginary frequency")
            return

        if all([freq > -50 for freq in self.imaginary_frequencies[1:]]):
            logger.warning(
                "Had small imaginary modes - not displacing along "
                "other modes"
            )
            return

        # There is more than one imaginary frequency. Will assume that the most
        # negative is the correct mode..
        for disp_magnitude, ext in zip([1, -1], ["_dis", "_dis2"]):
            logger.info(
                "Displacing along second imaginary mode to try and " "remove"
            )

            disp_ts: TransitionState = self.copy()
            disp_ts.atoms = displaced_species_along_mode(
                self, mode_number=7, disp_factor=disp_magnitude
            ).atoms

            disp_ts._run_opt_ts_calc(
                method=get_hmethod(), name_ext=name_ext + ext
            )

            if (
                self.imaginary_frequencies is not None
                and len(self.imaginary_frequencies) == 1
            ):
                logger.info(
                    "Displacement along second imaginary mode "
                    "successful. Now have 1 imaginary mode"
                )

                # Set the new properties of this TS from a successful reopt
                self.atoms = disp_ts.atoms
                self.energy = disp_ts.energy
                self._hess = disp_ts.hessian
                break

        return None

    def find_lowest_energy_ts_conformer(
        self, rmsd_threshold: Optional[float] = None
    ):
        """Find the lowest energy transition state conformer by performing
        constrained optimisations"""
        logger.info("Finding lowest energy TS conformer")
        assert self.energy is not None, "Must have a TS energy"

        # Generate a copy of this TS on which conformers are searched, for
        # easy reversion
        _ts: TransitionState = self.copy()
        _ts.hessian, _ts.gradient, _ts.energy = None, None, None

        hmethod = get_hmethod() if Config.hmethod_conformers else None
        _ts.find_lowest_energy_conformer(hmethod=hmethod)

        # Remove similar TS conformer that are similar to this TS based on root
        # mean squared differences in their structures being above a threshold
        rmsd_threshold = (
            Config.rmsd_threshold if rmsd_threshold is None else rmsd_threshold
        )
        _ts.conformers = Conformers(
            [
                conf
                for conf in _ts.conformers
                if calc_heavy_atom_rmsd(conf.atoms, self.atoms)
                > rmsd_threshold
            ]
        )

        logger.info(
            f"Generated {len(_ts.conformers)} unique (RMSD > "
            f"{rmsd_threshold} Å) TS conformer(s)"
        )

        if len(_ts.conformers) == 0:
            logger.info("Had no conformers - no need to re-optimise")
            return

        # Optimise the lowest energy conformer to a transition state - will
        # .find_lowest_energy_conformer will have updated self.atoms etc.
        _ts.optimise(name_ext="optts_conf")

        if _ts.is_true_ts and _ts.energy < self.energy:
            logger.info("Conformer search successful - setting new attributes")

            self.atoms = _ts.atoms
            self.energies = _ts.energies
            self._hess = _ts.hessian
            return None

        de = "nan" if _ts.energy is None else f"{_ts.energy - self.energy:.4f}"  # type: ignore
        logger.warning(
            f"Transition state conformer search failed "
            f"(∆E = {de} Ha). Reverting"
        )
        return None

    @property
    def is_true_ts(self) -> bool:
        """Is this TS a 'true' TS i.e. has at least on imaginary mode in the
        hessian and is the correct mode"""

        if self.energy is None:
            logger.warning("Cannot be true TS with no energy")
            return False

        if self.has_imaginary_frequencies and self.has_correct_imag_mode:
            logger.info(
                "Found a transition state with the correct "
                "imaginary mode & links reactants and products"
            )
            return True

        return False

    def save_ts_template(self, folder_path: Optional[str] = None) -> None:
        """Save a transition state template containing the active bond lengths,
         solvent and charge in folder_path

        -----------------------------------------------------------------------
        Keyword Arguments:
            folder_path (str): folder to save the TS template to

            (default: {None})
        """
        if self.bond_rearrangement is None:
            raise ValueError(
                "Cannot save a TS template without a bond " "rearrangement"
            )

        logger.info(f"Saving TS template for {self.name}")

        truncated_graph = get_truncated_active_mol_graph(self.graph)

        for bond in self.bond_rearrangement.all:
            truncated_graph.edges[bond]["distance"] = self.distance(*bond)

        ts_template = TStemplate(truncated_graph, species=self)
        ts_template.save(folder_path=folder_path)

        logger.info("Saved TS template")
        return None

    @classmethod
    def from_species(cls, species: "Species") -> "TransitionState":
        """
        Generate a TS from a species. Note this does not set the bond rearrangement
        thus mode checking will not work from this species.

        -----------------------------------------------------------------------
        Arguments:
            species:

        Returns:
            (autode.transition_states.transition_state.TransitionState): TS
        """
        return cls(ts_guess=TSguess.from_species(species), bond_rearr=None)
