Tutorial
========

All the examples here can be viewed at `GitHub <https://github.com/duartegroup/autodE/tree/master/example>`_. They have
been tested using XTB & ORCA electronic structure theory packages.


S\ :sub:`N`\2 Reaction
----------------------

.. image:: ../examples/common/sn2_image.png

Considering the S\ :sub:`N`\2 reaction between F- and methyl chloride in water we have the
smiles strings for the reactant and products generated from Chemdraw:

.. note::
    flouride        : [F-]

    methyl_chloride : CCl

    chloride        : [Cl-]

    methyl_flouride : CF

First import the required objects and set the number of cores

.. code-block:: python

  >>> from autode import Reactant, Product, Config, Reaction
  >>> Config.n_cores = 4


Initalise the reactants and products from their respective SMILES strings

.. code-block:: python

    >>> flouride = Reactant(name='F-', smiles='[F-]', solvent='water')
    >>> methyl_chloride = Reactant(name='CH3Cl', smiles='ClC', solvent='water')
    >>> chloride = Product(name='Cl-', smiles='[Cl-]', solvent='water')
    >>> methyl_flouride = Product(name='CH3F', smiles='CF', solvent='water')

Here a solvent is also specified, but this may be left unspecified for reactions in the gas phase. Then from the
reactants and products form a reaction and calculate the reaction profile.

.. code-block:: python

  >>> reaction = Reaction(flouride, methyl_chloride, chloride, methyl_flouride)
  >>> reaction.calculate_reaction_profile()

This function call will return a plot something like:

.. image:: ../examples/common/sn2_reaction_profile.png

where conformers of the reactant and products have been searched and the profile calculated at
PBE0-D3BJ/def2-TZVP//PBE0-D3BJ/def2-SVP. It should take around 5 minutes to complete on a modern processor.


Diels–Alder
-----------

For the most simple [4+2] Diels–Alder reaction between ethene and butadiene we have

.. code-block:: python

 >>> butadiene = Reactant(name='diene', smiles='C=CC=C')
 >>> ethene = Reactant(name='ethene', smiles='C=C')
 >>> cyclohexene = Product(name='cyclohexene', smiles='C1CC=CCC1')
 >>> da_reaction = Reaction(butadiene, ethene, cyclohexene, name='DA')
 >>> da_reaction.calculate_reaction_profile()

which generates

.. image:: ../examples/common/da_reaction_profile.png

where the reaciton barrier is smaller than would be expected as the entropic penalty for forming the highly organised
TS is not included.