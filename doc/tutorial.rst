Tutorial
========

All the examples here can be viewed at `GitHub <https://github.com/duartegroup/autodE/tree/master/examples>`_. They have
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

    >>> flouride = Reactant(name='F-', smiles='[F-]')
    >>> methyl_chloride = Reactant(name='CH3Cl', smiles='ClC')
    >>> chloride = Product(name='Cl-', smiles='[Cl-]')
    >>> methyl_flouride = Product(name='CH3F', smiles='CF')

Here a solvent is also specified, but this may be left unspecified for reactions in the gas phase. Then from the
reactants and products form a reaction and calculate the reaction profile.

.. code-block:: python

  >>> reaction = Reaction(flouride, methyl_chloride, chloride, methyl_flouride, solvent_name='water')
  >>> reaction.calculate_reaction_profile()

This function call will return a plot something like:

.. image:: ../examples/common/sn2_reaction_profile.png

where conformers of the reactant and products have been searched and the profile calculated at
PBE0-D3BJ/def2-TZVP//PBE0-D3BJ/def2-SVP. It should take around 5 minutes to complete on a modern processor.

