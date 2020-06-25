**********************
Molecular Manipulation
**********************

**autodE** provides some simple methods for molecular manipulation and more
functionality when combined with `molfunc <https://github.com/duartegroup/molfunc/>`_.

Fragmentation
-------------

From a molecular graph representation of a molecule its fragmentation is
relatively straightforward. For example, to fragment methane to CH\ :sub:`3`\ • +
H•

.. code-block:: python

  >>> from autode import Molecule
  >>> methane = Molecule(name='CH4', smiles='C')
  >>> methane.atoms
  [[C,  0.000,   0.004, -0.020],
   [H, -0.657,  -0.848, -0.321],
   [H, -0.458,   0.975, -0.306],
   [H,  0.085,  -0.025,  1.080],
   [H,  1.030,  -0.105, -0.432]]

.. code-block:: python

  >>> from autode.mol_graphs import split_mol_across_bond
  >>> ch3_nodes, h_nodes = split_mol_across_bond(methane.graph, bond=(0, 1))
  >>> ch3 = Molecule(name='CH3', mult=2, atoms=[methane.atoms[i] for i in ch3_nodes])
  >>> ch3.atoms
  [[C,  0.000,   0.004, -0.020],
   [H, -0.458,   0.975, -0.306],
   [H,  0.085,  -0.025,  1.080],
   [H,  1.030,  -0.105, -0.432]]
  >>> h = Molecule(name='H', mult=2, atoms=[methane.atoms[i] for i in h_nodes])
  >>> h.atoms
  [[H, -0.657,  -0.848, -0.321]]


Functionalisation
-----------------

.. image:: ../common/functionalisation.png

Swapping fragments on a structure (e.g. H → Me) can be achieved using SMILES
concatenation. For example to stitch two methyl fragments to generate an
ethane molecule

.. code-block:: python

  >>> ethane = Molecule(name='C2H6', smiles='C%99.C%99')
  >>> ethane.n_atoms
  8

Multiple fragments can be added to the same core by specifying multiple sites
on a single atom

.. code-block:: python

  >>> propane = Molecule(name='C3H8', smiles='C%99%98.C%99.C%98')
  >>> propane.n_atoms
  11

This method regenerates the whole structure which may not be desirable if the
molecule is a transition state (TS) or a particular conformation of interest.

molfunc
_______

Adding a fragment to a fixed core structure can be achieved with
`molfunc <https://github.com/duartegroup/molfunc/>`_ and can be installed
with: :code:`pip install molfunc`. **molfunc** requires a xyz file to
initialise a molecule and indexes atoms from 1 so that atom 2 is the first
hydrogen atom in methane

.. code-block:: python

  >>> from molfunc import CoreMolecule, CombinedMolecule
  >>> methane.print_xyz_file()
  >>> methane_core = CoreMolecule(xyz_filename='CH4.xyz', atoms_to_del=[2])
  >>> ethane = CombinedMolecule(methane_core, frag_smiles='C[*]', name='C2H6')
  >>> ethane.n_atoms
  8

A set of fragments can be iterated through using **molfunc** to generate a
library rapidly e.g.

.. literalinclude:: ../common/methane_molfunc.py

Out (visualised):

.. image:: ../common/molfunc_functionalisation.png
