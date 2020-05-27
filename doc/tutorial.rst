Tutorial
========

All the examples here can be viewed at `GitHub <https://github.com/duartegroup/autodE/tree/master/examples>`_. They have
been tested using XTB & ORCA electronic structure theory packages.

Installation Check
------------------

autodE will pick up any electronic structure theory packages with implemented wrappers (ORCA, NWChem, Gaussian09, XTB
and MOPAC) that are available from your *PATH* environment variable. To check the expected high and low level methods are
available:

.. code-block:: python

  >>> from autode import methods
  >>> methods.get_hmethod()
  <autode.wrappers.ORCA.ORCA object at XXXXXXXXXXX>
  >>> methods.get_lmethod()
  <autode.wrappers.XTB.XTB object object at XXXXXXXXXXX>


If autode.exceptions.MethodUnavailable is raised see the troubleshooting :doc:`page <troubleshooting>`.

S\ :sub:`N`\2 Reaction
----------------------

.. image:: ../examples/common/sn2_image.png

Considering the S\ :sub:`N`\2 reaction between F- and methyl chloride in water we have the
smiles strings for the reactant and products generated from Chemdraw (by selecting a molecule → Edit → Copy As → SMILES):

.. note::
    Fluoride : [F-]

    MeCl     : CCl

    Chloride : [Cl-]

    MeF      : CF

First, import the required objects and set the number of cores

.. code-block:: python

  >>> from autode import Reactant, Product, Config, Reaction
  >>> Config.n_cores = 4


Initialise reactants and products from their respective SMILES strings

.. code-block:: python

    >>> Fluoride = Reactant(name='F-', smiles='[F-]')
    >>> MeCl = Reactant(name='CH3Cl', smiles='ClC')
    >>> Chloride = Product(name='Cl-', smiles='[Cl-]')
    >>> MeF = Product(name='CH3F', smiles='CF')

Then, from reactants and products form a reaction in water and calculate the reaction profile.

.. code-block:: python

  >>> reaction = Reaction(Fluoride, MeCl, Chloride, MeF, name='sn2', solvent_name='water')
  >>> reaction.calculate_reaction_profile()

This function call will return a plot something like:

.. image:: ../examples/common/sn2_reaction_profile.png

as *sn2_reaction_profile.png* in the current working directory, where conformers of the reactant and products have been
searched and the profile calculated at PBE0-D3BJ/def2-TZVP//PBE0-D3BJ/def2-SVP using an implicit water solvent.
It should take around 10 minutes to complete on a modern processor.

