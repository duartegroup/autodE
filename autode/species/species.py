import numpy as np
import autode.values as val
from copy import deepcopy
from datetime import date
from typing import Optional, Union, List, Collection
from scipy.spatial import distance_matrix
from autode.log import logger
from autode.atoms import Atom, Atoms, AtomCollection
from autode.exceptions import CalculationException
from autode.geom import calc_rmsd
from autode.constraints import Constraints
from autode.log.methods import methods
from autode.conformers.conformers import Conformers
from autode.solvent.solvents import ExplicitSolvent, get_solvent
from autode.calculation import Calculation
from autode.wrappers.keywords import Keywords
from autode.config import Config
from autode.input_output import atoms_to_xyz_file
from autode.mol_graphs import is_isomorphic, reorder_nodes
from autode.methods import get_lmethod, get_hmethod, ElectronicStructureMethod
from autode.mol_graphs import make_graph
from autode.hessians import Hessian
from autode.units import ha_per_ang_sq, ha_per_ang
from autode.thermochemistry.symmetry import symmetry_number
from autode.thermochemistry.igm import calculate_thermo_cont, LFMethod
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
                  f'{self.name}, '
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

    def new_species(self, name='species') -> 'Species':
        """
        A new version of this species, identical properties without any
        energies/gradients/hessian/conformers

        Keyword Arguments:
            name (str): Name of the new species

        Returns:
            (autode.species.Species):
        """
        species = Species(name, deepcopy(self.atoms), self.charge, self.mult)
        species.graph = None if self.graph is None else self.graph.copy()
        species.solvent = None if self.solvent is None else self.solvent.copy()

        return species

    @property
    def charge(self) -> int:
        """Total charge on this species"""
        return self._charge

    @charge.setter
    def charge(self, value) -> None:
        self._charge = int(value)

    @property
    def mult(self) -> int:
        """Total spin multipliity on this species (2S + 1)"""
        return self._mult

    @mult.setter
    def mult(self, value) -> None:
        self._mult = int(value)

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
            self._hess = Hessian(value, atoms=self.atoms, units=ha_per_ang_sq)

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
            self._grad = val.Gradient(value, units=ha_per_ang)

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
        Vibrational frequencies, which are all but the lowest 6 for a
        non-linear molecule and all but the lowest 5 for a linear one

        Returns:
            (list(autode.value.Frequency) | None):
        """
        n = 6 if not self.is_linear() else 5

        return self.frequencies[n:] if self.frequencies is not None else None

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

        imag_freqs = [freq for freq in self.frequencies if freq.is_imaginary]

        if len(imag_freqs) == 0:
            logger.warning('No imaginary frequencies')
            return None

        return imag_freqs

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
    def energy(self, value: Union[val.Energy, float, None]):
        """
        Add an energy to the list of energies at this geometry

        Arguments:
            value (float | autode.values.Energy | None):
        """

        if value is None:
            # No change required
            pass

        elif isinstance(value, val.PotentialEnergy):
            self.energies.append(value)

        else:
            # Attempt to cast the value to Potential energy
            self.energies.append(val.PotentialEnergy(value))

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
            logger.warning('Could not calculate H - an energy was None')
            return None

    @property
    def n_conformers(self) -> int:
        """
        Number of conformers of this species

        Returns:
            (int):
        """
        return 0 if self.conformers is None else len(self.conformers)

    @property
    def conformers(self) -> Conformers:
        return self._conformers

    @conformers.setter
    def conformers(self,
                   value: Optional[List['autode.conformers.Conformer']]) -> None:
        """
        Set conformers of this species

        Arguments:
            value (list(autode.conformers.Conformer) | None):
        """
        if value is None:
            self._conformers.clear()
            return

        self._conformers = Conformers([conf for conf in value])

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
            if conformer.energy is None or conformer.atoms is None:
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

        if lowest_energy is None:
            raise RuntimeError("Failed to set the lowest energy conformer as "
                               "no suitable conformers were present")

        return None

    def populate_conformers(self, *args, **kwargs):
        """Populate self.conformers"""
        return self._generate_conformers(*args, **kwargs)

    @requires_atoms
    def reorder_atoms(self, mapping: dict) -> None:
        """
        Reorder the atoms in this species (in place) using a mapping. For
        example, to reorder the atoms in a HF molecule:

        .. code-block:: Python

            >>> import autode as ade
            >>> hf = ade.Species(name='HF', charge=0, mult=1,
            ...                   atoms=[ade.Atom('H'), ade.Atom('F', x=1)])
            >>> hf.atoms
            Atoms([Atom(H, 0.0 0.0 0.0), Atom(F, 1.0, 0.0, 0.0)])
            >>> hf.reorder_atoms(mapping={0: 1, 1: 0})


        Arguments:
            mapping (dict): Dictionary keyed with current atom indexes with
                            the values as the required indexing

        Raises:
            (ValueError): If the mapping is invalid
        """
        if not (set(mapping.keys()) == set(mapping.values()) == set(list(range(self.n_atoms)))):
            raise ValueError('Invalid mapping. Must be 1-1 for all atoms')

        self._atoms = Atoms([self.atoms[i] for i in
                             sorted(mapping, key=mapping.get)])

        if self.graph is None:
            return  # No need to re-order a graph that is not set

        self.graph = reorder_nodes(graph=self.graph,
                                   mapping={u: v for v, u in mapping.items()})
        return

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

            angle_tol (autode.values.Angle): Tolerance on the angle considered
                                             to be linear
        """
        if tol is not None:
            angle_tol = val.Angle(np.arccos(1.0 - tol), units='rad')

        return self.atoms.are_linear(angle_tol=angle_tol)

    @requires_atoms
    def translate(self, vec: Collection[float]) -> None:
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
                       title_line:            Optional[str] = None,
                       filename:              Optional[str] = None,
                       additional_title_line: Optional[str] = None) -> None:
        """
        Print a standard xyz file from this Molecule's atoms

        Keyword Arguments:
            title_line (str | None): String to add as the second line of the
                                     .xyz file

            filename (str | None): Filename ending with .xyz. If None then will
                                   use the name of this molecule

            additional_title_line (str | None): Additional elements to add to
                                                then title line
        """

        if filename is None:
            filename = f'{self.name}.xyz'

        # Default generated title line
        if title_line is None:
            title_line = f'Generated by autodE on: {date.today()}. '
            if self.energy is not None:
                title_line += f'E = {self.energy:.6f} Ha'

        if additional_title_line is not None:
            title_line += additional_title_line

        return atoms_to_xyz_file(self.atoms, filename, title_line=title_line)

    @requires_atoms
    def optimise(self,
                 method:      Optional[ElectronicStructureMethod] = None,
                 reset_graph: bool = False,
                 calc:        Optional[Calculation] = None,
                 keywords:    Optional[Keywords] = None,
                 n_cores:     Optional[int] = None) -> None:
        """
        Optimise the geometry using a method

        Arguments:
            method (autode.wrappers.base.ElectronicStructureMethod):

        Keyword Arguments:
            reset_graph (bool): Reset the molecular graph

            calc (autode.calculation.Calculation): Different e.g. constrained
                                                   optimisation calculation

            keywords (autode.wrappers.keywords.Keywords):

            n_cores (int | None): Number of cores to use for the calculation,
                                  if None then will default to
                                  autode.Config.n_cores
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
                               n_cores=Config.n_cores if n_cores is None else n_cores)

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
                    method:     Optional[ElectronicStructureMethod] = None,
                    calc:       Optional[Calculation] = None,
                    temp:       float = 298.15,
                    lfm_method: Union[LFMethod, str, None] = None,
                    ss:         Optional[str] = None,
                    **kwargs) -> None:
        """Calculate the free energy contribution for a species

        Keyword Arguments:
            method (autode.wrappers.base.ElectronicStructureMethod):

            calc (autode.calculation.Calculation):

            temp (float): Temperature in K

            lfm_method (LFMethod | str | None): Method to treat low freqency
                                           modes. {'igm', 'truhlar', 'grimme'}.
                                           Defaults to Config.lfm_method

            ss (str | None): Standard state to use.
                             Defaults to Config.standard_state

        Raises:
            (autode.exceptions.CalculationException | KeyError):

        See Also:
            (autode.thermochemistry.igm.calculate_thermo_cont)
        """
        if lfm_method is not None:
            kwargs['lfm_method'] = lfm_method

        if type(lfm_method) is str:
            try:
                kwargs['lfm_method'] = LFMethod[lfm_method.lower()]
            except KeyError:
                raise ValueError(f'{lfm_method} is not valid. Must be on of '
                                 f'{[m for m in LFMethod]}')

        if ss is not None:
            if ss.lower() not in ('1m', '1atm'):
                raise ValueError(f'{ss} is not a valid standard state. Must be'
                                 ' either "1m" or "1atm"')
            kwargs['ss'] = ss.lower()

        if calc is not None and calc.output.exists:
            self.atoms = calc.get_final_atoms()
            self.energy = calc.get_energy()
            self.hessian = calc.get_hessian()

        elif (calc is not None and not calc.output.exists
              or self.hessian is None):
            logger.info('Calculation did not exist or Hessian was None - '
                        'calculating the Hessian')
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
                     keywords: Optional[Keywords] = None,
                     n_cores:  Optional[int] = None) -> None:
        """Calculate the single point energy of the species with a
        autode.wrappers.base.ElectronicStructureMethod"""
        logger.info(f'Running single point energy evaluation of {self.name}')

        sp = Calculation(name=f'{self.name}_sp',
                         molecule=self,
                         method=method,
                         keywords=method.keywords.sp if keywords is None else keywords,
                         n_cores=Config.n_cores if n_cores is None else n_cores)
        sp.run()
        energy = sp.get_energy()

        if energy is None:
            raise CalculationException("Failed to calculate a single point "
                                       f"energy for {self}")
        self.energy = energy

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

        self.conformers.optimise(method=lmethod)
        self.conformers.prune(remove_no_energy=True)

        if hmethod is not None:

            if Config.hmethod_sp_conformers:
                # Use only single point energies on lmethod geometries
                assert hmethod.keywords.low_sp is not None
                self.conformers.single_point(method=hmethod,
                                             keywords=hmethod.keywords.low_sp)
            else:
                # Otherwise run a full optimisation
                self.conformers.optimise(hmethod)

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

        self._charge = int(charge)
        self._mult = int(mult)

        self.solvent = get_solvent(solvent_name=solvent_name)

        #: All energies calculated at a geometry (autode.values.Energies)
        self.energies = val.Energies()

        self._grad = None
        self._hess = None

        #: Molecular graph with atoms(V) and bonds(E) (NetworkX.Graph | None)
        self.graph = None

        self._conformers = Conformers()

        self.constraints = Constraints()
