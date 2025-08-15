Changelog
=========

1.4.6
------
-------

Bug Fixes
*********
- Fixes Molfunc example to run with current version

1.4.5
------
-------

Functionality improvements
**************************
- Adds a boolean option for calculating single points refinements
- AutodE can now handle any number of bond rearrangements
- Graph isomorphism check is faster, due to additional screening with isospectral method

Bug Fixes
*********
- Fixes coordinate extraction in some G16 output files
- Fixes loading old mlptrain .npz files
- Fixes compatability with RDKit 2025.03.2

Usability improvements/Changes
******************************
- Drops Python 3.8 support
- Catches conformer calculation exceptions
- Added :code:`Species.print_mol_file()` function that can print mol-files with connectivity from graph

1.4.4
------
-------

Functionality improvements
**************************
- Improved constrained optimisation (:code:`CRFOptimiser`) and handling of multiple constraints
- Adds compatability with numpy v2.0
- Improved implementation of the RFO-TRM (:code:`QAOptimiser`) optimiser that can handle constraints
- Added static internal back-transform and damping for faster and easier DIC to Cartesian coordinate transformation

Bug Fixes
*********
- DIC to Cartesian transform will now always use :code:`PIC.close_to()` to ensure steps along dihedral have the smallest change, even after back-transform is complete

Usability improvements/Changes
******************************
- Optimiser convergence criteria have been improved to consider energy change, RMS and max. gradient and step sizes.

1.4.3
------
-------

Functionality improvements
**************************
- DHS and DHS-GS can now switch between two step sizes
- Peak detection in bracket methods now uses cubic polynomial fit with energies and gradients instead of only projecting gradients

Bug Fixes
*********
- Fixes no solvent being added in QRC calculations
- Fixes cases where .xyz files are printed with no space between coordinates when the coordinate value is large.
- Fixes DHS and DHS-GS methods ignoring number of cores

Usability improvements/Changes
******************************
- Added consistent aliases for double dagger across all energies in :code:`autode.reaction.delta`
- Optimiser trajectories are saved on disk instead of keeping completely in memory
- Hessian updates the refactored into :code:`OptCoordinates` class

1.4.2
------
-------

Functionality improvements
**************************
- Replaces hard-coded derivatives for primitive internal coordinates with automatic differentiation
- More comprehensive primitive internal coordinate generation
- Adds the capacity to handle linear molecules in internal coordinate system

Bug Fixes
*********
- Fixes triangular rings being incorrectly treated as dihedral angles

Usability improvements/Changes
******************************
- Faster import of autode package by lazily loading matplotlib
- ORCA output files copied after a calculation are configurable

1.4.1
------
-------

Functionality improvements
**************************
- Adds the thermochemistry method from A. Otlyotov, Y. Minenkov in https://doi.org/10.1002/jcc.27129
- Adds the improved Elastic Image Pair (i-EIP) method for double-ended transition state search
- Adds a :code:`autode.Species.solvent_name` property and setter for setting solvents from a string
- Enables reloading molecules from xyz files with their previously defined charge/multiplicity/solvent

Bug Fixes
*********
- Fixes Hessian extraction in some G16 output files
- Fixes large step sizes in DHSGS
- Fixes the ability to define both a SMILES string and a xyz file without raising an exception


1.4.0
--------
----------

Functionality improvements
**************************
- Adds :code:`temporary_config()` context manager for temporary configuration changes
- Adds a :code:`ForceConstant` value
- Adds a more robust hybrid RFO-TRM geometry optimiser with trust radius update and a feature to detect oscillation and damp it
- Adds Dewar-Healy-Stewart (DHS) method and its variant DHS-GS to find transition states from initial and final geometries


Usability improvements/Changes
******************************
- Adds full usability of autodE on Windows, including parallelisation with :code:`loky`
- Optional timeout for graph isomorphism test in Windows, turned on by :code:`Config.use_experimental_timeout=True` (default behaviour kept for Linux/macOS)
- The electronic temperature and the version of parameterisation for xTB calculations are made configurable
- A NEB :code:`Image` now derives from a :code:`Species` superclass
- Modifies NEB :code:`Image` constructor to be formed from an image
- Defines named constructors (:code:`from_end_points(...)`, :code:`from_list(...)`) for :code:`NEB`
- Removes :code:`NEB().contains_peak()` in favour of :code:`NEB().images.contains_peak`
- Modifies the :code:`CImages` constructor to ensure it's constructed from an :code:`Images` instance
- Removes :code:`NEB.get_species_saddle_point()` in favour of :code:`NEB.peak_species`
- Defines named constructors (:code:`from_end_points(...)`, :code:`from_list(...)`) for :code:`NEB`
- Removes :code:`mol_graphs.get_atom_ids_sorted_type()`

Bug Fixes
*********
- Fixes pickling issue with :code:`autode.config.Config` on Windows and in multiprocessing spawn for Linux/macOS
- Fixes RFO Hessian update bug
- Fixes QChem not using :code:`Config.max_core`


1.3.5
--------
----------


Usability improvements/Changes
******************************
- :code:`autode.value.ValueArray.to()` now defaults to copying the object rather than inplace modification


Functionality improvements
**************************
- Adds a :code:`to_` method to :code:`autode.value.ValueArray` for explicit inplace modification of the array


Bug Fixes
*********
- Fixes :code:`ERROR` logging level being ignored from environment variable :code:`AUTODE_LOG_LEVEL`
- Fixes :code:`autode.values.Value` instances generating items with units on division, and throw a warning if multiplying
- Fixes the printing of cartesian constraints in XTB input files, meaning they are no longer ignored
- Fixes :code:`Hessian` instances changing units when normal modes are calculated
- Fixes an incorrect alias for :code:`ev_per_ang`


1.3.4
--------
----------

Feature additions.

Usability improvements/Changes
******************************
* Throw useful exception for invalid :code:`ade.Config.ts_template_folder_path`


Functionality improvements
**************************
- Adds :code:`ade.transition_states.TransitionState.from_species` method to construct transition states from a species or molecule
- Adds :code:`autode.Reaction.save()` and :code:`autode.Reaction.load()` to save and reload a reaction state
- Adds saving checkpoints of a reaction during :code:`autode.Reaction.calculate_reaction_profile`


Bug Fixes
*********
- Fixes calculation :code:`clean_up()` failing with a null filename


1.3.3
--------
----------

Bugfix release.


Functionality improvements
**************************
- Adds skipping Hessian re-evaluation when using autodE optimisers if a molecules has a Hessian calculated at the same level of theory
- Adds a Hessian recalculation frequency to :code:`autode.optimisers.PRFOptimiser`
- Improves the default step size for TS optimising to be consistent with the ORCA default

Bug Fixes
*********
- Adds checking of SMILES-defined charge against the user-specified value
- Fixes :code:`autode.optimisers.CRFOptimiser` building incomplete internal coordinates for partially or completely fragmented molecules


1.3.2
--------
----------

Bugfix release.


Usability improvements/Changes
******************************
* Removes :code:`autode.geom.get_distance_constraints` in favour of a better named method :code:`TSBase().active_bond_constraints`


Bug Fixes
*********
- :code:`autode.transition_states.ts_guess.TSguess.from_species` now inherits solvent from the species
- Fixes a possible race condition in I/O for XTB conformer optimisations


1.3.1
--------
----------

Bugfix release.


Bug Fixes
*********
- Fixes behaviour of :code:`autode.utils.work_in_tmp_dir` and :code:`autode.utils.work_in` decorators
- Fixes an exception being raised when :code:`autode.Calculation.clean_up` is called with a method that doesn't implement external I/O
- Fixes autodE driven optimisations skipping execution when the input but not name changes


1.3.0
--------
----------

Optimisation features, graph assignment improvements and bugfixes.


Usability improvements/Changes
******************************
* Defines dummy atoms to have zero covalent and vdW radii
* Renames :code:`Method().available` to :code:`Method().is_available`
* Removes :code:`autode.bonds.get_ideal_bond_length_matrix` and :code:`autode.bonds.get_avg_bond_length`
* Removes :code:`autode.geom.rotate_columns`
* Modifies the names of most optimiser classes e.g. :code:`autode.opt.optimisers.PRFOOptimiser` -> :code:`PRFOptimiser`
* Simplifies initialising a :code:`autode.calculations.Calculation` by:

  * Requiring constraints to be attributed to a molecule
  * Removing the :code:`bond_ids_to_add` argument and using the labeled graph instead (active edges)
  * Removing the :code:`other_input_block` argument and appending to the keywords instead

* Removes :code:`autode.calculations.Calculation.print_final_output_lines` in favour of a method on :code:`calculation.output`
* Makes many methods in :code:`autode.calculations.Calculation` private
* Deprecates all :code:`autode.calculations.Calculation.get_<method>` methods in favour of setting properties of the input molecule
* Returns :code:`None` rather than rasies exceptions when calling the (deprecated) calculation get methods, to be consistent with ...get_energy
* Adds an :code:`autode.wrappers.keywords` package to improve file structure
* Removes any exceptions on calling :code:`.run()` on an optimiser instance where the system has no degrees of freedom
* Removes support for Python < v3.8
* Tweaks the default ORCA TS optimisation keywords to be more conservative, i.e. slower and more accurate


Functionality improvements
**************************
- Adds a :code:`autode.atoms.Atom.covalent_radius` property
- Adds a :code:`autode.atoms.Atoms.eqm_bond_distance` method for the equilibrium bonded distance between two atoms
- Adds vibrational frequency scaling through both :code:`autode.Config.freq_scale_factor` and a default value in wrapped functional keywords
- Adds a *much* more robust constrained rational function constrained optimiser in delocalised internal coordinates (DIC)
- Adds bond angle and dihedral primitive coordinates which can form part of the DIC set
- Improves the back transformation
- Adds an optional callback argument to :code:`autode.opt.optimisers.base.Optimiser` for running custom functions after every optimisation step
- Adds the ability to save/reload an :code:`autode.opt.optimisers.NDOptimiser` instance to/from a file
- Adds a solvent attribute to a :code:`autode.transition_states.transition_state.TransitionState` constructor
- Adds functionality to partition a nudged elastic band into images where the maximum atom-atom distance between images is below a threshold
- Adds a sequential adapt+NEB TS finding method where a pure adapt. path fails to generate a geometry close enough to the TS for a successful TS optimisation


Bug Fixes
*********
- Fixes variable harmonic frequencies (<2 cm-1 differences) due to projection vectors becoming close to rotational axes
- Fixes the extraction of atomic partial charges from ORCA output files
- Fixes gradients and Hessians not being reset on a molecule where the coordinates change
- Fixes unhelpful exception when calculating thermochemistry with EST methods without implemented "get_hessian" methods


See the table below for a quick benchmark of constrained optimisations in autodE
compared to ORCA. In all cases the structures were generated from SMILES strings (RDKit)
and optimised with a single constraint on the (0,1) distance of +0.1 Å from its current
value.

.. list-table::
    :header-rows: 1

    * - Molecule
      - autodE
      - ORCA
    * - C7H12
      - 6
      - 7
    * - C3H7
      - 8
      - 17
    * - C4H6
      - 3
      - 5
    * - CClH3
      - 3
      - 4
    * - C2H3O2
      - 3
      - 7
    * - C2FH5
      - 3
      - 5
    * - C4H6O2S
      - 6
      - 11

1.2.3
--------
----------

Minor functionality improvements and bugfixes.


Usability improvements/Changes
******************************
- All exceptions now inherit from a base :code:`autode.exceptions.AutodeException`
- Fixes a typo in :code:`autode.exceptions.UnsupoportedCalculationInput`
- Adds documentation explaining the intention of each exception in  :code:`autode.exceptions`
- Molecular graphs are now 'laziliy-loaded' i.e. generated once when the property is accessed


Functionality improvements
**************************
- Adds the ability to define atom classes for molecules in turn allowing for identity reactions to be calculated


Bug Fixes
*********
- Fixes clashing names for a reaction initialised explicitly from molecules without defined names


1.2.2
--------
----------

Bugfix release.


Bug Fixes
*********
- Fixes output redirection from XTB calculations resulting in missed lines on Mac


1.2.1
--------
----------

Bugfix release.


Functionality improvements
******************************
- Adds :code:`autode.mol_graphs.MolecularGraph` (subclass of :code:`networkx.Graph`) with a :code:`expected_planar_geometry` method
- Adds an :code:`are_planar` method to :code:`autode.atoms.Atoms`


Bug Fixes
*********

- Fixes a bug in assigning reasonable geometries which, in turn, could lead to skipped single point energy evaluations


1.2.0
--------
----------

Adds optimisation algorithms experimental explicit solvation, improves potential energy surface
module as well as an array of usability improvements.


Usability improvements/Changes
******************************
- Adds more type hints and documentation
- Updates the TS template saved in the default template library
- Adds a setter for :code:`autode.species.Species.solvent` so :code:`mol.solvent = 'water'` will assign a :code:`autode.solvent.Solvent`
- Removes :code:`autode.calculation.CalculationInput.solvent` as an attribute in favour of using the molecule's solvent
- Removes :code:`autode.calculation.get_solvent_name` in favour of a molecule check
- Removes :code:`autode.species.molecule.reactant_to_product` in favour of a :code:`to_product()` method for :code:`autode.species.molecule.Reactant` (and likewise with a Reactant)
- Removes partially implemented :code:`autode.species.molecule.SolvatedMolecule` and :code:`autode.species.complex.SolvatedReactantComplex` as the type of solvation (implicit/explicit) should be a property of the solvent and not the molecule
- Removes :code:`autode.reactions.Reaction.calc_deltaXXX` in favour of :code:`autode.reactions.Reaction.delta()`
- Refactors classes to place constructors at the top
- Removes :code:`autode.values.PlottedEnergy` as an estimated attribute is useful for all energies, not just those that are plotted
- Removes :code:`autode.reactions.Reaction.find_lowest_energy_ts` as the function is not well named and can be replaced by a :code:`autode.reactions.Reaction.ts` property
- Adds :code:`autode.transition_states.TransitionStates` as a wrapper for TSs, much like :code:`autode.conformers.Conformers`
- Updates :code:`autode.solvent.solvents.get_solvent` to require specifying either an implicit or explicit solvent
- Improves validation of distance constraints and adds invariance to the key order i.e. :code:`autode.constraints.distance[(0, 1)] == autode.constraints.distance[(1, 0)]`
- Removes :code:`autode.KcalMol` and :code:`KjMol` and enables a reaction to be plotted using a string representation of the units.
- Allows for keywords to be set using just a list or a string, rather than requiring a specific type
- Changes :code:`autode.wrappers.keywords.Keyword.has_only_name` to a property
- Modifies the constructor of :code:`autode.species.molecule.Molecule` to allow for a name to be specified when initialising from a .xyz file
- Modifies :code:`autode.calculation.Calculation.get_energy` to raise an exception if the energy cannot be extracted
- Adds a runtime error if e.g. :code:`autode.calculation.Calculation.get_energy` is called on a calculation that has not been run
- Skips low-level adaptive path searching if the high and low-level methods are identical (when XTB or MOPAC are not installed)
- Adds a default set of low-level single point keywords
- Adds a flag to override exiting a reaction profile calculation when association complexes are generated
- Adds a check that a calculation isn't going to exceed the maximum amount of physical memory on the computer


Functionality improvements
**************************

- Adds a selection of molecule optimisers to locate minima and transition states
- Refactors :code:`autode.smiles.angles` to use unique class names (preventing overlap with e.g. :code:`autode.values.Angle`)
- Adds a :code:`autode.solvent.Solvent.dielectric` property for a solvent's dielectric constant
- Adds a :code:`autode.solvent.Solvent.is_implicit` property
- Adds methods (e.g. translate and rotate) to :code:`autode.point_charges.PointCharge`
- Adds checking that both high and low-level electronic structure methods are available before running :code:`autode.reaction.Reaction.calculate_reaction_profile` or :code:`calculate_reaction_profile`
- Adds a more robust explicit solvation generation (:code:`autode.species.molecule.Molecule.explicitly_solvate()`)
- Removes criteria on using a TS template with large distance differences between the structure and the template in favour of running sequential constrained optimisations to the required point
- Rewrites :code:`autode.pes` into a consistent module while maintaining much of the functionality. Simplifies the interface
- Adds a QChem electronic structure method wrapper
- Adds :code:`autode.species.Species.calc_hessian` to calculate either an analytic or numerical Hessian (in parallel)
- Adds image dependent pair potential (IDPP) relaxation improved interpolated geometries
- Adds :code:`autode.hessians.HybridHessianCalculator` to calculate numerical Hessians at two levels of theory


Bug Fixes
*********

- Updates the TS template saved in the default template library
- Reloads output file lines from a failed then re-run calculation
- Fixes Hessian extractions from some Gaussian output files


1.1.3
--------
----------

Usability improvements

Usability improvements/Changes
******************************
- Improves consistency and behaviour of :code:`calc_thermo` method of a species, allowing for keywords and non-run calculations
- Allows for a non-fork multiprocessing 'start_method'


1.1.2
--------
----------

Bugfixes

Usability improvements/Changes
******************************
- Fixes typo in :code:`autode.exceptions.ReactionFormationFalied`

Bug Fixes
*********

- Fixes a bug where rings containing mostly double bonds failed to build with :code:`autode.smiles.builder.Builder`
- Fixes using XTB as a high-level method with the xtb-gaussian wrapper (thanks @kjelljorner)


1.1.1
--------
----------

Documentation and typing hints

Usability improvements/Changes
******************************
- Adds `typing <https://docs.python.org/3/library/typing.html>`_ to user-facing functions
- Adds :code:`autode.config.location` to easily locate the core configuration file for permanent editing
- Updates documentation for readability
- Ensures units are kept if constructing a :code:`Value` from a :code:`Value` (i.e. :code:`Value(x)`, when :code:`x` is a :code:`Value`)


Functionality improvements
**************************

- Changes :code:`Keyword` to an abstract base class
- Improves speed of :code:`Species` rotation (numpy rather than a Python for loop)


Bug Fixes
*********

- Fixes bug where NCI conformers were generated with the same name thus did not optimise uniquely (introduced in v.1.1.0)


1.1.0
--------
----------

API improvements that broadly maintain backwards compatibility.


Usability improvements/Changes
******************************
- Adds more argument and return types
- Changes :code:`AtomCollection.atoms` to a property for more flexible sub-classing
- Changes :code:`ElectronicStructureMethod.doi_str` and :code:`Keyword.doi_str` to properties
- Adds interpretable :code:`repr(Species)`
- :code:`Species.energies` is zeroed when the :code:`Species.atoms` are reset or change
- :code:`Species.energy` is a property of the last computed energy on that species
- :code:`Species.is_linear` now uses an angle tolerance to determine linearity, which is slightly tighter than the previous float-based tolerance
- Removes :code:`CalculationOutput.set_lines` in favour of a cached file_lines property to avoid :code:`set_file_lines()`
- Removes :code:`CalculationOutput.get_free_energy()` in favour of :code:`Species.free_energy` once a Hessian is set for a molecule and similarly with :code:`CalculationOutput.get_enthalpy()`
- Removes :code:`CalculationOutput.get_imaginary_freqs()` (now :code:`Species.imaginary_frequencies`) and :code:`CalculationOutput.get_normal_mode_displacements()` (now :code:`Species.normal_mode()`)
- :code:`Species.imaginary_frequencies` now returns :code:`None` rather than an empty list for a species without any imaginary frequencies, to be consistent with other properties
- Changes :code:`CalculationOutput.terminated_normally()` to a property (:code:`CalculationOutput.terminated_normally`)
- Removes :code:`Reaction.find_complexes` in favour of setting the reactant and product complexes dynamically, unless :code:`Reaction.calculate_complexes` is called to find association complexes
- Tweaks the default relative tolerance on bonds to account for M-X agostic interactions lengthening bonds
- Enables :code:`Species.atoms` to be added, even if they are `None`
- Improved atom setting of :code:`Complex.atoms`
- Changes :code:`Complex.get_atom_indexes()` to :code:`Complex.atom_indexes()`
- Changes :code:`Complex.molecules` to a private attribute as the atoms/energy/gradient is not propagated
- Allows for :code:`Species.translate()` and :code:`Species.rotate()` to be called using vectors as lists or tuples rather than just numpy arrays
- Modifies :code:`get_truncated_complex()` to :code:`get_truncated_species()` and changes the return type to a species to reflect a possibly different molecular composition of the complex
- Improves peak checking in adaptive path TS guess generation
- Removes :code:`autode.atoms.get_thing()` functions, in favour of :code:`Atom.thing`
- Raises an exception if a single point energy evaluation fails to execute successfully
- Removes :code:`autode.conformers.conformer.get_conformer()` in favour of a more flexible :code:`autode.conformer.Conformer` constructor
- Adds :code:`Species.constraints` that are used in optimisations (still available in :code:`Calculation` initialisation)
- Adds :code:`Conformers` to enable parallel electronic structure calculations across a set of conformers
- Improves readability of pruning of conformers based on RMSD and energy thresholds


Functionality improvements
**************************

- Adds angle and dihedral angle properties to an :code:`AtomCollection`
- Improves and adds more :code:`Unit` definitions
- Adds :code:`Value` and :code:`ValueArray` base classes for energies, gradients etc. These allow for implicit (1 Hartree == 617.509 kcal mol-1) comparisons and explicit conversion (1 Hartree).to('kcal')
- Adds further conversion factors to :code:`Constants`
- Adds :code:`Species.energies` as a container of all energies that have been calculated at a geometry
- Adds :code:`Keywords.bstring` as a 'brief' summary of the keywords e.g. PBE0/def2-SVP and are associated with an :code:`Energy` (a type of :code:`Value`)
- Improves quick reaction coordinate characterisation of TSs by providing a maximum atomic displacement for improved initial structures
- Adds Hessian diagonalisation to obtain normal modes with and without translation and rotation projections for linear and non-linear molecules
- Adds :code:`Species.weight` and :code:`Species.mass` as equivalent properties for the molecular weight
- Improves dihedral sampling in molecule generation
- Adds :code:`atoms.remove_dummy()` to remove all dummy atoms from a set
- Enables different force constants to be used in XTB constrained optimisations (:code:`Config.XTB.force_constant`, which sets :code:`wrappers.XTB.XTB.force_constant`)
- Adds :code:`Solvent.copy()`
- Adds :code:`Species.reorder_atoms()` to reorder the atoms in a species using a mapping
- Adds :code:`Config.ORCA.other_input_block` to allow for a block of input to be printed in all ORCA input files
- Changes the loose optimisations to only use a maximum of 10 iterations. This is based on an analysis of 3500 ORCA
optimisations, which plateaus quickly:

.. image:: common/opt_convergence_3500_ORCA.png
   :width: 500

suggesting a value of 10 is a appropriate. This will be system dependent and need increasing for
large/flexible systems. For path optimisations loose optimisations use a maximum of 50 cycles.


Bug Fixes
*********

- Skips conformers with no atoms in finding unique conformers
- Corrects benchmark TS location for the Grubbs metathesis example, where the reactant complex is bound
- Fixes possible zero distance constraint for a single atom
- Fixes spin state definition for XTB calculations
- Fixes possible override of a constructor-defined spin state by the SMILES parser


1.0.5
--------
----------

Bugfix release

Bug Fixes
*********
- Saves transition state templates with correct atom labels


1.0.4
--------
----------

Bug fixes in SMILES parser and 3D geometry builder from 1.0.3.


Usability improvements
**********************

- Improves doc strings
- Throws interpretable error when calling :code:`find_tss`  without :code:`reaction.reactant` set

Functionality improvements
**************************

- SMILES strings with >9 ring closures are parsed correctly
- cis-double bonds in rings no longer minimise with constraints, which is a little faster

Bug Fixes
*********
- Tweaks repulsion parameters in minimisation to build fused rings
- Enables SMILES parsing with "X(...)1" branching
- Fixes spin multiplicity for odd numbers of hydrogens
- Improves ring closure 3D build
- Fixes incorrect implicit valency for aromatic heteroatoms
- Improves metal finding in SMILES strings with regex
- Corrects atom type for sp2 group 16 elements
- Fixes dihedral rotation with atoms not close to any other


1.0.3
--------
----------

A minor API revision from 1.0.2 but adds C++ extension which should be extensible to
further developments of fast C-based code.

Usability improvements
**********************

- :code:`autode.Species()` inherit from a :code:`AtomCollection()` base class for more flexibility

- :code:`autode.Constants` attributes have more readable names (while retaining backwards compatability)

- :code:`autode.geom.length()` as an explicit alias of :code:`np.linalg.norm` has been removed

- :code:`autode.input_output.xyz_file_to_atoms()` throws more informative errors

- :code:`autode.mol_graphs.make_graph()` throws NoAtomsInMolecule for a species with no atoms

- :code:`species.formula` and :code:`species.is_explicitly_solvated` are now a properties

- :code:`autode.smiles.parser` has been rewritten & is (hopefully) a more robust SMILES parser


Functionality improvements
**************************

- Metal complex initial geometries can now be generated with the correct stereochemistry

- Macrocycles default to an **autodE** builder that conserves SMILES stereochemistry (`RDKit#1852 <https://github.com/rdkit/rdkit/issues/1852>`_)

- :code:`species.coordinates` can be set from either 3xN matrices or 3N length vectors

- :code:`autode.Atom()`s have :code:`.group` :code:`.period` and :code:`.tm_row` properties referring to their location in the periodic table

- :code:`autode.atoms.PeriodicTable` added

- :code:`species.bond_matrix` added as a property and returns a boolean array for interactions between all atom pairs


Bug Fixes
*********

- :code:`reaction.calculate_complexes()` calls :code:`reaction.find_complexes()` if needed thus can be called in isolation



1.0.2
--------
----------

Usability improvements
**********************

- Effective core potentials can now be specified in :code:`Keywords()`

- ORCA fitting basis sets now default to def2/J, which should be smaller but as accurate as AutoAux

- Molecule initialisation from a .xyz file now checks for an odd number of electrons. For example, :code:`Molecule('H_atom.xyz')` will raise a :code:`ValueError` but :code:`Molecule('H_atom.xyz', charge=1)` or :code:`Molecule('H_atom.xyz', mult=2)` are acceptable


Functionality improvements
**************************

- :code:`atom.atomic_number` has been added as an atom attribute

- :code:`atom.atomic_symbol` is a more intuitive alias for :code:`atom.label`



1.0.1
--------
------------


Usability improvements
**********************

- Molecular complexes can now be initialised with a reasonable geometry :code:`Complex(..., do_init_translation=True)`


Functionality improvements
**************************

- :code:`species.radius` has been added as an approximate molecular radius (in Angstroms, excluding VdW radii)


Bug Fixes
*********

- Final breaking bond distances are now the minimum of the product X-Y distance if present in the product, or 2x the distance. This is required for breaking bonds that cross a ring.

- Neighbour lists for comparing possibly equivalent bond rearrangements are now compared using a sorted list


1.0.0
--------
------------

The first stable release! Mostly documentation updates from v.1.0.0b3 with the
package now being conda-install-able.


Usability improvements
**********************

- More documentation


Functionality improvements
**************************

- XTB wrapper now supports v. 6.4 (and hopefully higher)


Thanks to Joe, Alistair, Matina, Kjell, Gabe, Cher-Tian amongst others for their invaluable contributions.


1.0.0b3
--------
------------

This version brings several major changes and in some instances breaks
backwards compatibility, but does feature significant improvements in speed
and accuracy for finding transition states.

Usability improvements
**********************

- :code:`species.get_distance(i, j)` is now :code:`species.distance(i, j)`

- :code:`species.set_atoms(new_atoms)`  is now properly handled with a setter so :code:`species.atoms = new_atoms` will set the new atoms

- :code:`species.n_atoms` is more robust

- :code:`species.get_coordinates()` is now :code:`species.coordinates`, returning a numpy array copy of the species coordinates (Nx3 in Å)

- :code:`species.centre()` will translate a species so it's coordinate centroid lies at the origin

- PBE0/def2-SVP is now the default 'low opt' method (`keywords.low_opt`) with loose optimisation. Path exploration uses this method, thus it needs to be very close to the 'opt' level


Functionality improvements
**************************

- 1D, 2D potential energy surface scans and nudged elastic band (NEB) methods to generate TS guesses from reactants have been replaced by an adaptive path search which seems to be very efficient for generating initial paths
For the prototypical SN2 between fluoride and methyl chloride the relaxed PES (PBE0-D3BJ/ma-def2-SVP/CPCM(water)) is


.. image:: common/adapt_surface_sn2.png
   :width: 500

where the previously employed linear path (red) is compared to the adaptive scheme (blue, purple) and the 'true' intrinsic reaction coordinate.
With a small minimum step size a path very close to the MEP is traversed with a very small number of required constrained optimisations. This
enables NEB relaxations to be skipped and the associated limitations (corner cutting, oscillating path, optimisation in Cartesian coordinates)
avoided. This exploration is essential when a linear path over multiple bonds leads to rearrangements, e.g. an (E2) elimination reaction the
comparison for the linear, adaptive and IRC paths are shown below


- (CI)-NEB with adaptive force constant has been added

- Initial path exploration from reactants is performed at the 'low_opt' level with a final breaking bond distance below.

Previous implementations made use of a 1.5 Å additional shift for uncharged reactions
and 2.5 Å for charged, this however lead to possible final C-H distances of ~3.6 Å and steps
into unphysical regions. 1.0.0b3 uses an estimate based on the distance where the bond
is mostly broken, as below


.. image:: common/XY_bde_XTB.png

where X-Y corresponds to a molecule e.g. C-C with the appropriate hydrogens added
then the BDE curve calculated at the GFN2-XTB level of theory. A multiplier of ~2 affords a
'mostly broken bond' (i.e. the distance at 3/4 of energy of the broken bond).

- There is now a heuristic used to skip TSs that go via small rings (3, 4-membered) if there is a >4-membered equivalent (:code:`ade.Config.skip_small_ring_tss`)


Bug Fixes
*********

- Calculations are now unique based on constraints, so NEB calculations executed in the same directory are not skipped with different bond rearrangements
