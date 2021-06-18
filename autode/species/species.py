import numpy as np
import autode.values as val
from copy import deepcopy
from typing import Optional, Union, List
from scipy.spatial import distance_matrix
from autode.log import logger
from autode.atoms import Atom, Atoms, AtomCollection
from autode.geom import calc_rmsd
from autode.log.methods import methods
from autode.conformers.conformers import get_unique_confs
from autode.solvent.solvents import ExplicitSolvent, get_solvent
from autode.calculation import Calculation
from autode.wrappers.keywords import Keywords
from autode.config import Config
from autode.input_output import atoms_to_xyz_file
from autode.mol_graphs import is_isomorphic
from autode.conformers.conformers import conf_is_unique_rmsd
from autode.methods import get_lmethod, get_hmethod, ElectronicStructureMethod
from autode.mol_graphs import make_graph
from autode.hessians import Hessian
from autode.thermochemistry.symmetry import symmetry_number
from autode.thermochemistry.igm import calculate_thermo_cont
from autode.utils import (requires_atoms,
                          work_in,
                          requires_conformers)


class Species(AtomCollection):

    def __str__(self):
        """Unique species identifier"""

        if self.atoms is None:
            atoms_str = ''

        else:
            # Only use the first 100 atoms
            atoms_str = ''.join([atom.label for atom in self.atoms[:100]])

        solv_str = self.solvent.name if self.solvent is not None else 'none'

        return f'{self.name}_{self.charge}_{self.mult}_{atoms_str}_{solv_str}'

    def _repr(self, prefix: str):
        """Base representation of a Species/Molecule/Complex etc."""

        string = (f'{prefix}('
                  f'n_atoms={self.n_atoms}, '
                  f'charge={self.charge}, '
                  f'mult={self.mult})')

        return string

    def __repr__(self):
        """Brief representation of this species"""
        return self._repr(prefix='Species')

    def copy(self) -> 'Species':
        """Copy this whole species"""
        return deepcopy(self)

    @AtomCollection.atoms.setter
    def atoms(self,
              value: Union[List[Atom], Atoms, None]):
        """
        Set the atoms for this species, and reset the energies

        Arguments:
            value (list(autode.atoms.Atom) | None):
        """

        if value is None:
            self._atoms = None
            return

        # If the geometry is identical up to rotations/translations then
        # energies do not need to be changed
        if (self.n_atoms == len(value)
            and all(a.label == v.label for a, v in zip(self.atoms, value))):

            rmsd = calc_rmsd(coords1=np.array([v.coord for v in value]),
                             coords2=np.array([a.coord for a in self.atoms]))
        else:
            rmsd = None

        if rmsd is None or rmsd > 1E-8:
            logger.info(f'Geometry changed- resetting energies of {self.name}')
            self.energies.clear()
            self.gradient = None
            self.hessian = None

        self._atoms = Atoms(value)
        return

    @property
    def formula(self) -> str:
        """Return the molecular formula of this species, e.g.::

            self.atoms = None                 ->   ""
            self.atoms = [Atom(H), Atom(H)]   ->  "H2"

        Returns:
            (str):
        """

        if self.atoms is None:
            return ""

        symbols = [atom.label for atom in self.atoms]

        formula_str = ''
        for symbol in sorted(set(symbols)):
            num = symbols.count(symbol)
            formula_str += f'{symbol}{num if num > 1 else ""}'

        return formula_str

    @property
    def hessian(self) -> Optional[Hessian]:
        """
        Hessian (d^2E/dx^2) at this geometry (autode.values.Hessian | None)
        shape = (3*n_atoms, 3*n_atoms)
        """
        return self._hess

    @hessian.setter
    def hessian(self,
                value: Union[Hessian, np.ndarray, None]):
        """Set the Hessian matrix as a Hessian value"""

        req_shape = (3*self.n_atoms, 3*self.n_atoms)
        if hasattr(value, 'shape') and value.shape != req_shape:
            raise ValueError('Could not set the Hessian. Incorrect shape: '
                             f'{value.shape} != {req_shape}')

        if value is None:
            self._hess = None

        elif isinstance(value, Hessian):
            self._hess = value

            if self._hess.atoms is None:
                self._hess.atoms = self.atoms

        elif isinstance(value, np.ndarray):
            logger.warning('Setting the Hessian from a numpy array - assuming '
                           'units of Ha Å^-2')
            self._hess = Hessian(value, atoms=self.atoms, units='Ha/ang^2')

        else:
            raise ValueError(f'Could not set Hessian with {value}, Must be '
                             f'a numpy array or a Hessian.')

    @property
    def gradient(self) -> Optional[val.Gradient]:
        """
        Gradient (dE/dx) at this geometry (autode.values.Gradients | None)
        shape = (n_atoms, 3)
        """
        return self._grad

    @gradient.setter
    def gradient(self,
                 value: Union[val.Gradient, np.ndarray, None]):
        """Set the gradient matrix"""

        if hasattr(value, 'shape') and value.shape != (self.n_atoms, 3):
            raise ValueError('Could not set the gradient. Incorrect shape: '
                             f'{value.shape} != {(self.n_atoms, 3)}')

        if value is None:
            self._grad = None

        elif isinstance(value, val.Gradient):
            self._grad = value

        elif isinstance(value, np.ndarray):
            logger.warning('Setting the gradients from a numpy array - '
                           'assuming Ha / Å units')
            self._grad = val.Gradient(value, units='Ha/ang')

        else:
            raise ValueError(f'Could not set the gradient with {value}, Must '
                             f'be a numpy array or a Hessian.')

    @property
    def frequencies(self) -> Optional[List[val.Frequency]]:
        """
        Frequencies from Hessian diagonalisation, in cm-1 by default and
        are projected from rotation and translation

        Returns:
            (list(autode.values.Frequency) | None):
        """
        if self.hessian is None:
            logger.warning('No Hessian has been calculated - no frequencies')
            return None

        return self.hessian.frequencies_proj

    @property
    def vib_frequencies(self) -> Optional[List[val.Frequency]]:
        """
        Vibrational frequencies, which are all but the lowest 6

        Returns:
            (list(autode.value.Frequency) | None):
        """
        return self.frequencies[6:] if self.frequencies is not None else None

    @property
    def imaginary_frequencies(self) -> Optional[List[val.Frequency]]:
        """
        Imaginary frequencies of a molecule

        Returns:
            (list(autode.values.Frequency) | None):
        """
        if self.frequencies is None:
            logger.warning('Had no frequencies - could not find any imaginary')
            return None

        return [freq for freq in self.frequencies if freq.is_imaginary]

    def normal_mode(self, mode_number: int) -> Optional[val.Coordinates]:
        """
        Vibrational normal mode indexed from 0, the first 6 are translation
        and rotation and have zero displacements. The first vibrational mode
        has mode_number = 6.

        Arguments:
            mode_number (int):

        Returns:
            (autode.values.Coordinates):
        """
        if self.hessian is None:
            logger.warning('Could not calculate a normal mode displacement')
            return None

        return self.hessian.normal_modes_proj[mode_number]

    @property
    @requires_atoms
    def bond_matrix(self) -> np.ndarray:
        """Return a np.ndarray boolean array of the bonds

        Returns:
            (np.ndarray): shape = (n_atoms, n_atoms)

        Raises:
             (ValueError): If the molecular graph is nor set
        """

        matrix = np.zeros(shape=(self.n_atoms, self.n_atoms), dtype=bool)

        if self.graph is None:
            raise ValueError('No molecular graph set. Bonds are not defined')

        for bond in self.graph.edges:
            matrix[tuple(bond)] = matrix[tuple(reversed(bond))] = True

        return matrix

    @property
    def radius(self) -> val.Distance:
        """Calculate an approximate radius of this species"""
        if self.n_atoms == 0:
            return val.Distance(0.0)

        coords = self.coordinates
        return val.Distance(np.max(distance_matrix(coords, coords)) / 2.0)

    @property
    def sn(self) -> int:
        """
        Calculate the symmetry number (σ_R) of the atoms. Only implemented for
        'small' molecules <50 atoms


        References:
        [1] Theor Chem Account (2007) 118:813
        [2] . Phys. Chem. B (2010) 114:16304

        Returns:
            (int): σ_R
        """
        if self.n_atoms == 0:
            return 1

        if self.n_atoms > 50:
            logger.warning('Symmetry number calculations are not implemented '
                           'for large molecules. Assuming C1 -> σ_R=1')
            return 1

        return symmetry_number(self)

    @property
    def is_explicitly_solvated(self) -> bool:
        return isinstance(self.solvent, ExplicitSolvent)

    @property
    def energy(self) -> Optional[val.PotentialEnergy]:
        """
        Last computed potential energy

        Returns:
            (autode.values.PotentialEnergy):
        """
        return self.energies.last(val.PotentialEnergy)

    @energy.setter
    def energy(self, value: Optional[val.Energy]):
        """
        Add an energy to the list

        Arguments:
            value (float | autode.values.Energy):
        """

        if value is not None:
            energy = val.PotentialEnergy(value)
            self.energies.append(energy)

    @property
    def h_cont(self) -> Optional[val.EnthalpyCont]:
        """
        Return the enthalpic contribution to the energy

        Returns:
             (autode.values.Energy | None): H - E_elec
        """
        return self.energies.last(val.EnthalpyCont)

    @property
    def g_cont(self) -> Optional[val.FreeEnergyCont]:
        """
        Return the Gibbs (free) contribution to the energy

        Returns:
             (autode.values.Energy | None): G - E_elec
        """
        return self.energies.last(val.FreeEnergyCont)

    @property
    def free_energy(self) -> Optional[val.FreeEnergy]:
        """
        Free energy (G or A) of this species, calculated using the last energy
        and free energy contribution

        Returns:
            (autode.values.FreeEnergy | None):
        """
        try:
            return val.FreeEnergy(self.energy + self.g_cont)

        except TypeError:
            logger.warning('Could not calculate G - an energy was None')
            return None

    @property
    def enthalpy(self) -> Optional[val.Enthalpy]:
        """
        Enthalpy (H) of this species, calculated using the last energy and
        enthalpy contribution

        Returns:
            (autode.values.FreeEnergy | None):
        """
        try:
            return val.Enthalpy(self.energy + self.h_cont)

        except TypeError:
            logger.warning('Could not calculate the H - an energy was None')
            return None

    def _set_unique_conformers_rmsd(self, conformers, n_sigma=5):
        """
        Given a list of conformers add those that are unique based on an RMSD
        tolerance. In addition, discard any very high or very low energy
        conformers more than n_sigma σ (std. devs) away from the average

        Args:
            conformers (Iterable(autode.conformers.Conformer):

            n_sigma (int): Number of standard deviations a conformer energy
                           must be from the average for it not to be added
        """
        energies = std_dev_e = avg_e = None
        self.conformers = []

        # Populate an array of energies in any units to calculate std. dev. etc
        if all(conf.energy is not None for conf in conformers):
            energies = np.array([conf.energy for conf in conformers])
            std_dev_e, avg_e = np.std(energies), np.average(energies)

            logger.warning(f'Have {len(energies)} energies with μ={avg_e:.6f} '
                           f'σ={std_dev_e:.6f}')

        for i, conf in enumerate(conformers):

            if energies is not None:

                if np.abs(conf.energy - avg_e) < 1E-8 and len(self.conformers) > 0:
                    logger.warning(f'Conformer {i} had an identical energy'
                                   f' - not adding')
                    continue

                if np.abs(conf.energy - avg_e)/max(std_dev_e, 1E-8) > n_sigma:
                    logger.warning(f'Conformer {i} had an energy >{n_sigma}σ '
                                   f'from the average - not adding')
                    continue

            if conf_is_unique_rmsd(conf, self.conformers):
                if self.graph is None:
                    logger.warning('Setting a conformer without a graph')
                else:
                    conf.graph = self.graph.copy()

                self.conformers.append(conf)

        logger.info(f'Generated {len(self.conformers)} unique conformer(s)')
        return None

    def _generate_conformers(self, *args, **kwargs):
        raise NotImplementedError('Could not generate conformers. '
                                  'generate_conformers() not implemented')

    def _run_hess_calculation(self, method):
        """Run a Hessian calculation on this species"""

        if self.n_atoms < 2:
            logger.warning(f'Not running a Hessian calculation on only '
                           f'{self.n_atoms} atom(s). Cannot have frequencies')
            return None

        method = method if method is not None else get_hmethod()

        calc = Calculation(name=f'{self.name}_hess',
                           molecule=self,
                           method=method,
                           keywords=method.keywords.hess,
                           n_cores=Config.n_cores)
        calc.run()
        self.energy = calc.get_energy()
        self.hessian = calc.get_hessian()

        return None

    @requires_conformers
    def _set_lowest_energy_conformer(self):
        """Set the species energy and atoms as those of the lowest energy
        conformer"""
        lowest_energy = None

        for conformer in self.conformers:
            if conformer.energy is None:
                continue

            # Conformers don't have a molecular graph, so make it
            make_graph(conformer)

            if not is_isomorphic(conformer.graph, self.graph,
                                 ignore_active_bonds=True):
                logger.warning('Conformer had a different graph. Ignoring')
                continue

            # If the conformer retains the same connectivity, up the the active
            # atoms in the species graph

            if lowest_energy is None:
                lowest_energy = conformer.energy

            if conformer.energy <= lowest_energy:
                self.atoms = conformer.atoms
                self.energy = conformer.energy
                lowest_energy = conformer.energy

        return None

    @requires_atoms
    def is_linear(self,
                  tol:       Optional[float] = None,
                  angle_tol: val.Angle = val.Angle(1.0, units='deg')) -> bool:
        """
        Determine if a species is linear i.e all atoms are colinear

        Keyword Arguments:
            tol (float | None): Tolerance on |cos(θ)| - 1 where θ is the angle
                                between the vector from atom 0 to 1 and from
                                0 to n (n > 1). Present for compatibility and
                                overrides angle_tol if not None

            angle_tol (Angle): Tolerance on the angle considered to be linear
        """
        if tol is not None:
            angle_tol = val.Angle(np.arccos(1.0 - tol), units='rad')

        return self.atoms.are_linear(angle_tol=angle_tol)

    @requires_atoms
    def translate(self, vec: np.ndarray) -> None:
        """Translate the molecule by vector

        Arguments:
            vec (np.ndarray): shape = (3,)
        """
        for atom in self.atoms:
            atom.translate(vec)

        return None

    @requires_atoms
    def rotate(self,
               axis:   np.ndarray,
               theta:  float,
               origin: Optional[np.ndarray] = None) -> None:
        """Rotate the molecule by around an axis

        Arguments:
            axis (np.ndarray): Axis to rotate around. shape = (3,)

            theta (float): Angle to rotate anticlockwise by (radians)

        Keyword Arguments:
            origin (np.ndarray | None): Origin of the rotation
        """
        for atom in self.atoms:
            atom.rotate(axis, theta, origin=origin)

        return None

    @requires_atoms
    def centre(self) -> None:
        """Translate this molecule so the centroid (~COM) is at the origin"""
        self.translate(vec=-np.average(self.coordinates, axis=0))
        return None

    @requires_atoms
    def print_xyz_file(self,
                       title_line: str = '',
                       filename:   Optional[str] = None) -> None:
        """Print a standard xyz file from the Molecule's atoms"""

        if filename is None:
            filename = f'{self.name}.xyz'

        return atoms_to_xyz_file(self.atoms, filename, title_line=title_line)

    @requires_atoms
    def optimise(self,
                 method:      Optional[ElectronicStructureMethod] = None,
                 reset_graph: bool = False,
                 calc:        Optional[Calculation] = None,
                 keywords:    Optional[Keywords] = None) -> None:
        """
        Optimise the geometry using a method

        Arguments:
            method (autode.wrappers.base.ElectronicStructureMethod):

        Keyword Arguments:
            reset_graph (bool): Reset the molecular graph

            calc (autode.calculation.Calculation): Different e.g. constrained
                                                   optimisation calculation

            keywords (autode.wrappers.keywords.Keywords):

        Raises:
            (autode.exceptions.CalculationException):
        """
        logger.info(f'Running optimisation of {self.name}')

        if calc is None and method is None:
            raise ValueError('Optimisation cannot be performed without '
                             'a specified method or calculation.')

        if calc is None:
            calc = Calculation(name=f'{self.name}_opt',
                               molecule=self,
                               method=method,
                               keywords=method.keywords.opt if keywords is None else keywords,
                               n_cores=Config.n_cores)

        calc.run()
        self.atoms = calc.get_final_atoms()
        self.energy = calc.get_energy()

        method_name = '' if method is None else method.name
        self.print_xyz_file(filename=f'{self.name}_optimised_{method_name}.xyz')

        if reset_graph:
            make_graph(self)

        return None

    @requires_atoms
    def calc_thermo(self,
                    method: Optional[ElectronicStructureMethod] = None,
                    calc:   Optional[Calculation] = None,
                    temp:   float = 298.15,
                    **kwargs) -> None:
        """Calculate the free energy contribution for a species

        Keyword Arguments:
            method (autode.wrappers.base.ElectronicStructureMethod):

            calc (autode.calculation.Calculation):

            temp (float): Temperature in K

        Raises:
            (autode.exceptions.CalculationException):

        See Also:
            (autode.thermo.igm.calculate_thermo_cont)
        """

        if calc is not None and calc.output.exists:
            self.atoms = calc.get_final_atoms()
            self.energy = calc.get_energy()
            self.hessian = calc.get_hessian()

        elif (calc is not None and not calc.output.exists
              or self.hessian is None):
            self._run_hess_calculation(method=method)

        calculate_thermo_cont(self, temp=temp, **kwargs)
        return None

    @requires_atoms
    def calc_g_cont(self, *args, **kwargs) -> None:
        """Calculate the Gibbs free (G) contribution for this species  using
        Species.calc_thermo()"""
        return self.calc_thermo(*args, **kwargs)

    @requires_atoms
    def calc_h_cont(self, *args, **kwargs) -> None:
        """Calculate the enthalpic (H) contribution for this species using
        Species.calc_thermo()"""
        return self.calc_thermo(*args, **kwargs)

    @requires_atoms
    def single_point(self,
                     method:   ElectronicStructureMethod,
                     keywords: Optional[Keywords] = None) -> None:
        """Calculate the single point energy of the species with a
        autode.wrappers.base.ElectronicStructureMethod"""
        logger.info(f'Running single point energy evaluation of {self.name}')
        keywords = method.keywords.sp if keywords is None else keywords

        sp = Calculation(name=f'{self.name}_sp', molecule=self, method=method,
                         keywords=keywords, n_cores=Config.n_cores)
        sp.run()
        self.energy = sp.get_energy()

        return None

    @work_in('conformers')
    def find_lowest_energy_conformer(self,
                                     lmethod: Optional[ElectronicStructureMethod] = None,
                                     hmethod: Optional[ElectronicStructureMethod] = None) -> None:
        """
        For a molecule object find the lowest conformer in energy and set the
        molecule.atoms and molecule.energy

        Arguments:
            lmethod (autode.wrappers.ElectronicStructureMethod):
            hmethod (autode.wrappers.ElectronicStructureMethod):
        """
        logger.info('Finding lowest energy conformer')

        if self.n_atoms <= 2:
            logger.warning('Cannot have conformers of a species with 2 atoms '
                           'or fewer')
            return None

        if lmethod is None:
            logger.info('Getting the default low level method')
            lmethod = get_lmethod()

        methods.add('Low energy conformers located with the')
        self._generate_conformers()

        # For all generated conformers optimise with the low level of theory
        method_string = f'and optimised using {lmethod.name}'
        if hmethod is not None:
            method_string += f' then with {hmethod.name}'
        methods.add(f'{method_string}.')

        for conformer in self.conformers:
            conformer.optimise(lmethod)

        # Strip conformers that are similar based on an energy criteria or
        # don't have an energy
        self.conformers = get_unique_confs(conformers=self.conformers)

        if hmethod is not None:
            # Re-evaluate the energy of all the conformers with the higher
            # level of theory
            for conformer in self.conformers:

                if Config.hmethod_sp_conformers:
                    assert hmethod.keywords.low_sp is not None
                    conformer.single_point(hmethod)

                else:
                    # Otherwise run a full optimisation
                    conformer.optimise(hmethod)

        self._set_lowest_energy_conformer()

        logger.info(f'Lowest energy conformer found. E = {self.energy}')
        return None

    # --- Method aliases ---
    symmetry_number = sn

    def __init__(self,
                 name:         str,
                 atoms:        Union[List[Atom], Atoms, None],
                 charge:       Union[float, int],
                 mult:         Union[float, int],
                 solvent_name: Optional[str] = None):
        """
        A molecular species. A collection of atoms with a charge and spin
        multiplicity in a solvent (None is gas phase)

        ----------------------------------------------------------------------
        Arguments:
            name (str): Name of the species

            atoms (list(autode.atoms.Atom)): List of atoms in the species,
                                             or None

            charge (int): Charge on the species

            mult (int): Spin multiplicity of the species. 2S+1, where S is the
                        number of unpaired electrons

        Keyword Arguments:
            solvent_name (str | None): Name of the solvent_name, or None for a
                                       species  in the gas phase
        """
        super().__init__(atoms=atoms)

        self.name = name

        self.charge = int(charge)
        self.mult = int(mult)

        self.solvent = get_solvent(solvent_name=solvent_name)

        #: All energies calculated at a geometry (autode.values.Energies)
        self.energies = val.Energies()
        self._grad = None
        self._hess = None

        #: Molecular graph with atoms(V) and bonds(E) (NetworkX.Graph | None)
        self.graph = None

        # Conformers pf this species (list(autode.conformers.Conformer) | None)
        self.conformers = None
