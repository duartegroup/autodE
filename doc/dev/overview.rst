**************
Code Structure
**************


Overview
########

The **autodE** code base is structured around a :code:`Reaction` class, due to
the initial singular goal of calculating reaction profiles.

.. image:: ../common/reaction_simple_uml.svg
   :target: ../_images/reaction_simple_uml.svg
   :width: 550
   :align: center

|
|

The key attributes include :code:`reacs` and :code:`prods` as individual molecules
comprising the reactants and products of the reaction. Transition states connecting
reactants and products are held in :code:`tss`. From these, property attributes are generated,
including :code:`reactant` and :code:`product`, which are the corresponding
association complexes of all reactants and products. The :code:`ts` property is
simply the lowest energy transition state and :code:`None` if
:code:`len(reaction.tss) == 0`.


Overall class structure
#######################

Zooming out, the composition and inheritance between of some of the
classes arising from :code:`Reaction` (center) is shown below.

|

.. image:: ../common/simple_uml.svg
  :target: ../_images/simple_uml.svg

|

Species
*******

Individual atoms are collected into a :code:`Atoms` class which becomes an
(effective) attribute of an :code:`AtomCollection`, which serves as a base
class for all objects with associated :code:`atoms`. A :code:`Species` adds
a molecular graph, solvent and name attributes and is the parent class of
a :code:`Molecule`, :code:`Complex` and :code:`TransitionState`.

Values
******

Quantities with associated units e.g. an angle or energy are subclasses of
:code:`autode.values.Value`, which facilitate the conversion between units
(using :code:`value.to()`). For example, :code:`species.energy` returns a
:code:`PotentialEnergy` instance (if the energy has been calculated with e.g.
:code:`species.single_point()`).


Solvent
*******

Species in the gas phase have :code:`species.solvent == None` while solvated
ones have an instance of either :code:`ImplicitSolvent` or :code:`ExplicitSolvent`,
inheriting from the base :code:`autode.solvent.Solvent` class. Explicit solvent
contains defined atoms, thus inherits also from :code:`AtomCollecion`.


Calculation
***********

Energies and derivatives thereof are obtained but running calculations using
external QM packages (e.g. Gaussian, etc.) through a :code:`Calculation` instance.
A :code:`Calculation` is initialised with a :code:`Species`, using a
:code:`Method` and :code:`Keywords` describing the type of calculation to
perform. Please reach out via `email <mailto:autodE-gh@outlook.com?subject=autodE%20EST%20method>`_ or slack to add a a new method.


calculate_reaction_profile
**************************

From the description of a reaction (e.g SMILES strings), **autodE** can generate
the reaction profile. It starts by building a :class:`Reaction` instance,
describing the reactants/products involved (see :class:`Molecule`) as well as
the transition states (see :code:`TransitionStates`). Calling the
:code:`calculate_reaction_profile` method on the reaction instance first locates
the lowest energy conformers of each reactant and product
(:code:`species.find_lowest_energy_conformer()`). The intermediate optimisations
are performed using a :code:`Calculation` instance, which is responsible for calling
a specified QM package (e.g. XTB or Gaussian) with :code:`calculation.run()`.
The generated output is then parsed and the output available from the calculation
instance e.g. :code:`calculation.get_final_atoms()`. A :code:`Calculation`
is constructed with a :code:`Method`, which serves as the QM wrapper. From
optimised reactants and products a transition state (TS) search is performed
by constructing association complexes of reactants and products, then searching
over bond additions and deletions to traverse a reasonable path. Once the
:code:`TransitionStates` instance has been populated the lowest is selected to
perform a conformer search. If required, the conformational space of the
:code:`ReactantComplex` and :code:`ProductComplex` attributes of the reaction
are optimised. Hessian calculations are performed if the thermochemical contributions
to the energy are required, followed by single-point energy evaluations on the
final geometries.
