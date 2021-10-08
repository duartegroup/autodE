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

  >>> import autode as ade
  >>> methane = ade.Molecule(smiles='C')
  >>> [atom.atomic_symbol for atom in methane.atoms]
  ['C', 'H', 'H', 'H', 'H']

.. code-block:: python

  >>> from autode.mol_graphs import split_mol_across_bond
  >>> ch3_nodes, h_nodes = split_mol_across_bond(methane.graph, bond=(0, 1))
  >>> ch3 = ade.Molecule(name='CH3', mult=2, atoms=[methane.atoms[i] for i in ch3_nodes])
  >>> ch3.atoms
  Atoms(n_atoms=4, [Atom(C,  0.0009,  0.0041, -0.0202),
                    Atom(H, -0.4585,  0.9752, -0.3061),
                    Atom(H,  0.0853, -0.0253,  1.0804),
                    Atom(H,  1.0300, -0.1058, -0.4327)])
  >>> h = ade.Molecule(name='H', mult=2, atoms=[methane.atoms[i] for i in h_nodes])
  >>> h.atoms
  Atoms(n_atoms=1, [Atom(H, -0.6577, -0.8481, -0.3214)])


Functionalisation
-----------------

.. image:: ../common/functionalisation.png

Swapping fragments on a structure (e.g. H → Me) can be achieved using SMILES
concatenation. For example to stitch two methyl fragments to generate an
ethane molecule

.. code-block:: python

  >>> ethane = ade.Molecule(name='C2H6', smiles='C%99.C%99')
  >>> ethane.n_atoms
  8

Multiple fragments can be added to the same core by specifying multiple sites
on a single atom

.. code-block:: python

  >>> propane = ade.Molecule(name='C3H8', smiles='C%99%98.C%99.C%98')
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
