Install
=======

Dependencies
------------
**autodE** is a Python package that relies on external electronic structure codes and requires:

- `Python <https://www.python.org/>`_ > v. 3.6

- One of:

  + `ORCA <https://sites.google.com/site/orcainputlibrary/home/>`_ > v. 4.0
  + `Gaussian09 <https://gaussian.com/glossary/g09/>`_
  + `Gaussian16 <https://gaussian.com/gaussian16/>`_
  + `NWChem <http://www.nwchem-sw.org/index.php/Main_Page>`_ > v. 6.6
  + `QChem <https://www.q-chem.com/>`_ > 5.4

- One of:

  + `XTB <https://www.chemie.uni-bonn.de/pctc/mulliken-center/software/xtb/xtb/>`_ > v. 6.1
  + `MOPAC <http://openmopac.net/>`_ v. 2016


Python dependencies listed `here <https://github.com/duartegroup/autodE/blob/master/requirements.txt>`_ are best satisfied using conda
(`anaconda <https://www.anaconda.com/distribution>`_ or `miniconda <https://docs.conda.io/en/latest/miniconda.html>`_);
the following guide assumes a conda install.

******

Conda: Mac OSX / Linux
----------------------

**autodE** is available through `conda <https://anaconda.org/conda-forge/autode>`_ and can be installed with::

    $ conda install autode --channel conda-forge


If the environment cannot be solved see `here <https://duartegroup.github.io/autodE/troubleshooting.html#conda-solve-fails>`_.
A Linux installation tutorial is available `here <https://youtu.be/ZUweT1Sc02s>`_.

******

Git: Mac OSX / Linux
--------------------

To build from source first clone the repository and ``cd`` there::

    $ git clone https://github.com/duartegroup/autodE.git
    $ cd autodE


then, install the appropriate dependencies (you may want to create a new `virtual
environment <https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html>`_) and install::

    $ conda install --file requirements.txt --channel conda-forge
    $ pip install .


.. note::
    A working C++ compiler supporting C++11 is required. Tested with clang and gcc.

Git: Windows
------------

In a bash shell within Windows Subsystem for Linux follow the steps above.

.. warning::
    Windows installation is only supported within Windows Subsystem for Linux (`WSL <https://docs.microsoft.com/en-us/windows/wsl/install-win10>`_)

******

Installation Check
------------------

**autodE** will find any electronic structure theory packages with implemented
wrappers (ORCA, NWChem, Gaussian, XTB and MOPAC) that are available from your
`PATH <https://en.wikipedia.org/wiki/PATH_(variable)>`_ environment variable.
To check the expected high and low level methods are available:

.. code-block:: python

  >>> import autode as ade
  >>> ade.methods.get_hmethod()
  ORCA(available = True)
  >>> ade.methods.get_lmethod()
  XTB(available = True)


If a :code:`MethodUnavailable` exception is raised see the :doc:`troubleshooting page <troubleshooting>`.
If **autodE** cannot be imported please open a issue on `GitHub <https://github.com/duartegroup/autodE/issues>`_.

******

Quick EST Test
--------------

If the high and/or low level electronic structure methods have been installed
for the first time, it may be useful to check they're installed correctly.
To run a quick optimisation of H\ :sub:`2`\:

.. code-block:: python

  >>> import autode as ade
  >>> h2 = ade.Molecule(smiles='[H][H]')
  >>> h2.optimise(method=ade.methods.get_lmethod())
  >>> h2.optimise(method=ade.methods.get_hmethod())
  >>> h2.energy
  Energy(-1.16401 Ha)
  >>> h2.atoms
  Atoms([Atom(H, 0.3805, 0.0000, 0.0000), Atom(H, -0.3805, 0.0000, 0.0000)])


If an :code:`AtomsNotFound` exception is raised it is likely that the electronic structure
package is not correctly installed correctly.

.. note::
    Calculations are performed on 4 CPU cores by default, thus the high and
    low-level methods must be installed as their parallel versions where
    appropriate.
