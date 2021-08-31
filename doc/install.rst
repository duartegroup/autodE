Install
=======

Dependencies
------------
**autodE** is a Python package that relies on external QM codes and requires:

- `Python <https://www.python.org/>`_ > v. 3.5

- One of:

  + `ORCA <https://sites.google.com/site/orcainputlibrary/home/>`_ ≥ v. 4.1
  + `Gaussian09 <https://gaussian.com/glossary/g09/>`_
  + `Gaussian16 <https://gaussian.com/gaussian16/>`_
  + `NWChem <http://www.nwchem-sw.org/index.php/Main_Page>`_ v. 6.6

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


******

Git: Mac OSX / Linux
--------------------

To build from source first clone the repository and ``cd`` there::

    $ git clone https://github.com/duartegroup/autodE.git
    $ cd autodE


then, install the appropriate dependencies (you may want to create a new virtual environment) and install::

    $ conda install --file requirements.txt --channel conda-forge
    $ python setup.py install


.. note::
    A working C++ compiler is required.

Setup video tutorial
********************

A Linux installation tutorial is available through the following link: https://youtu.be/ZUweT1Sc02s


Git: Windows
------------

.. warning::
    Windows installation is not currently supported, but *should* be possible through Windows Subsystem for Linux (`WSL <https://docs.microsoft.com/en-us/windows/wsl/install-win10>`_)

******

Installation Check
------------------

**autodE** will find any electronic structure theory packages with implemented wrappers (ORCA, NWChem, Gaussian09, XTB
and MOPAC) that are available from your *PATH* environment variable. To check the expected high and low level methods are
available:

.. code-block:: python

  >>> from autode import methods
  >>> methods.get_hmethod()
  <autode.wrappers.ORCA.ORCA object at XXXXXXXXXXX>
  >>> methods.get_lmethod()
  <autode.wrappers.XTB.XTB object object at XXXXXXXXXXX>


If a MethodUnavailable exception is raised see the :doc:`troubleshooting page <troubleshooting>`.

******

Quick test
----------

If the high and/or low level methods have been installed for the first time
it may be useful to check they're installed correctly. To run a quick optimisation
of H\ :sub:`2`\:

.. code-block:: python

  >>> import autode as ade
  >>> h2 = ade.Molecule(smiles='[H][H]')
  >>> h2.optimise(method=ade.methods.get_lmethod())
  >>> h2.optimise(method=ade.methods.get_hmethod())
  >>> h2.energy
  Energy(-1.16401 Ha)
  >>> h2.atoms
  [[H, 0.3805, 0.0000, 0.0000], [H, -0.3805, 0.0000, 0.0000]]

If an :code:`AtomsNotFound` exception is raised it is likely that the electronic structure
package is not correctly installed. Note that calculations are performed on 4 CPU cores
by default, thus the high and low-level methods must be installed as their parallel versions
where appropriate.
