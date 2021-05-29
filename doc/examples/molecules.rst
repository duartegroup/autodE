*********
Molecules
*********

Reactants and Products are :ref:`Molecules <molecules>` and are initialised
much like their :ref:`Species <species>` parent, but have charge and
multiplicity defaults (0, 1 respectively) and can be built from
`SMILES <https://en.wikipedia.org/wiki/Simplified_molecular-input_line-entry_system/>`_
strings

.. code-block:: python

  >>> from autode import Molecule
  >>> molecule = Molecule(name='molecule')
  >>> molecule.charge
  0
  >>> molecule.mult
  1

or from 3D structures given as xyz files directly

  >>> ch4 = Molecule('methane.xyz')
  >>> ch4.name
  'methane'

---------------

Simple Example
--------------

.. image:: ../common/water.png

To generate a water Molecule from its SMILES string ('O'), where hydrogen atoms
are implied

.. code-block:: python

  >>> water = Molecule(name='h2o', smiles='O')
  >>> water.atoms
  [[O, -0.00, 0.36, -0.00], [H, -0.86, -0.18, -0.00], [H, 0.83, -0.18, 0.00]]

Molecules also add a molecular graph atribute as a NetworkX `Graph <https://networkx.github.io/documentation/stable/reference/introduction.html#graphs/>`_
object and contain node (atoms) and edge (bonds) attributes

.. code-block:: python

  >>> water.graph
  <networkx.classes.graph.Graph object at XXXXXXXXX>
  >>> water.graph.nodes
  NodeView((0, 1, 2))
  >>> water.graph.edges
  EdgeView([(0, 1), (0, 2)])

where in water there are three atoms {0, 1, 2} and two bonds. The 3D structure
can be generated as a .xyz file for viewing in molecular visualisation software
(Avogadro, Chimera, VMD, Mercury, Maestro etc.)

.. code-block:: python

  >>> water.print_xyz_file()

where 'h2o.xyz' is generated in the current working directory.

---------------

Geometry Manipulation
---------------------

.. image:: ../common/water_shift.png

Whole molecules can be translated and rotated. For example, to translate the
water molecule so the oxygen atom is centred at the origin

.. code-block:: python

  >>> water.coordinates
  array([[-0.0011,  0.3631, -0.    ],
         [-0.825 , -0.1819, -0.    ],
         [ 0.8261, -0.1812,  0.    ]])
  >>> o_atom = water.atoms[0]
  >>> water.translate(vec=-o_atom.coord)
  >>> water.coordinates
  array([[ 0.    ,  0.    ,  0.    ],
         [-0.8250, -0.1819,  0.    ],
         [ 0.8261, -0.1812,  0.    ]])

then rotate around the x axis

.. code-block:: python

  >>> import numpy as np
  >>> water.rotate(axis=np.array([1.0, 0.0, 0.0]), theta=np.pi)
  >>> water.coordinates
  array([[ 0.    ,  0.   ,  0.    ],
         [-0.8250, 0.1819,  0.    ],
         [ 0.8261, 0.1812,  0.    ]])

Angles between atoms in a molecule can be also calculated

.. code-block:: python

  >>> water.angle(1, 0, 2)
  Angle(1.9752 rad)

where atoms are indexed from 0, so the angle is θ(H-O-H). As with distances,
explicit unit conversion is supported

.. code-block:: python

  >>> water.angle(1, 0, 2).to('deg')
  Angle(113.17085 °)



Calculations
------------

.. image:: ../common/water_opt_energy.png

**autodE** provides wrappers around common electronic structure theory packages
(ORCA, XTB, NWChem, MOPAC, Gaussian09) so geometries may be optimised and
energies calculated. Energies are in atomic Hartrees and gradients in
Ha / Å.

For example, to optimise the geometry at the XTB level and then perform a
single point energy evaluation with ORCA

.. code-block:: python

  >>> from autode.methods import XTB, ORCA
  >>> water.optimise(method=XTB())
  >>> water.energy
  Energy(-5.07054 Ha)
  >>> water.single_point(method=ORCA())
  >>> water.energy
  Energy(-76.37766 Ha)

where the default single point method in ORCA is PBE0-D3BJ/def2-TZVP. Like with
other values (distances, angles, dihedrals) converting to different units is as
simple as

.. code-block:: python

  >>> water.energy.to('kcal')
  Energy(-47927.6682 kcal mol-1)

The :code:`water.energy` returns the most recently evaluated energy at this geometry,
but the XTB energy is still saved as code:`water.energies`. Printing the energies
along with their associated methods

.. code-block:: python

  >>> for energy in water.energies:
  ...     print(energy, energy.method_str)
  ...
  Energy(-5.07054 Ha) xtb
  Energy(-76.37766 Ha) orca PBE0-D3BJ/def2-TZVP


Modifying the method is possible by setting the keywords

.. code-block:: python

  >>> from autode import SinglePointKeywords
  >>> orca = ORCA()
  >>> orca.keywords.sp = SinglePointKeywords(['PBE0', 'D3BJ', 'ma-def2-TZVP'])
  >>> water.single_point(method=orca)
  >>> water.energy
  -76.379214493975

.. note::
    Structure optimisation resets the positions of the atoms to their optimised
    value.
