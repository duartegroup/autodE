********************
Conformer Generation
********************

**autodE** generates conformers using two methods: (1)
`ETKDGv2 <https://doi.org/10.1021/acs.jcim.5b00654/>`_ implemented in
`RDKit <https://www.rdkit.org/>`_ and (2) a randomize & relax (RR) algorithm.


Butane
------

To generate conformers of butane initialised from a SMILES string defaults to
using ETKDGv2. The molecule's conformers are a list of
:ref:`Conformer <conformer>` objects, a subclass of :ref:`Species <species>`.

.. code-block:: python

  >>> from autode import Molecule
  >>> butane = Molecule(name='butane', smiles='CCCC')
  >>> butane.populate_conformers(n_confs=10)
  >>> len(butane.conformers)
  2

where although 10 conformers requested only two are generated. This because
by default there is an RMSD threshold used to remove identical conformers. To
adjust this threshold

.. code-block:: python

  >>> from autode import Config
  >>> Config.rmsd_threshold = 0.01
  >>> butane.populate_conformers(n_confs=10)
  >>> len(butane.conformers)
  8

For organic molecules ETKDGv2 is highly recommended while for metal
complexes the RR algorithm is used by default. To use RR for butane

.. code-block:: python

  >>> butane.rdkit_conf_gen_is_fine = False
  >>> butane.populate_conformers(n_confs=10)
  >>> for conformer in butane.conformers:
  ...     conformer.print_xyz_file()

Out (visualised)

.. image:: ../common/conformers.png

.. note::
   RMSD used by the RR algorithm applies to all atoms and does not account for
   symmetry (e.g. methyl rotation)


Metal Complex
-------------

.. image:: ../common/vaskas.png

Arbitrary distance constraints can be added in a RR conformer generation. For
example, to generate conformers of
`Vaska's complex <https://en.wikipedia.org/wiki/Vaska%27s_complex>`_
while retaining the square planar geometry


.. literalinclude:: ../common/vaskas_conformers.py

Out (visualised)

.. image:: ../common/vaskas_conformers.png

