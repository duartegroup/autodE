Install
=======

Dependencies
------------

* `Python <https://www.python.org/>`_ > v. 3.5
* One of:

  * `ORCA <https://sites.google.com/site/orcainputlibrary/home/>`_ > v. 4.1
  * `Gaussian09 <https://gaussian.com/glossary/g09/>`_
  * `Gaussian16 <https://gaussian.com/gaussian16/>`_
  * `NWChem <http://www.nwchem-sw.org/index.php/Main_Page>`_
* One of:

  * `XTB <https://www.chemie.uni-bonn.de/pctc/mulliken-center/software/xtb/xtb/>`_ > v. 6.1
  * `MOPAC <http://openmopac.net/>`_ v. 2016


The Python dependencies are listed in requirements.txt best satisfied using conda
(`anaconda <https://www.anaconda.com/distribution>`_ or `miniconda <https://docs.conda.io/en/latest/miniconda.html>`_);
the following guide assumes a conda install.

Mac OSX / Linux
---------------

First clone the repository and ``cd`` there::

    $ git clone https://github.com/duartegroup/autodE.git
    $ cd autodE


then, install the appropriate dependencies (you may want to create a new virtual environment)::

    $ conda config --append channels conda-forge --append channels omnia
    $ conda install --file requirements.txt

finally::

    $ python setup.py install


Setup video tutorial
***************

A Linux installation tutorial is available through the following link: https://youtu.be/ZUweT1Sc02s


Windows
--------

On Windows without a ``git`` installation **autode** can be installed with `anaconda <https://www.anaconda.com/distribution>`_
by: on the GitHub `page <https://github.com/duartegroup/autode>`_ using Clone or download → Download ZIP then
extracting it. Then, open an anaconda command prompt and ``cd`` to the directory and proceed as above e.g.::

    $ cd Downloads/autodE-master/
    $ conda config --append channels conda-forge
    $ conda install --file requirements.txt
    $ python setup.py install
.. note::
    The above commands assume you have extracted the zip to ``C:\Users\yourusername\Downloads`` and a C++
    compiler e.g. `VS <https://visualstudio.microsoft.com/vs/features/cplusplus/>`_ is available.

Installation Check
------------------

**autodE** will pick up any electronic structure theory packages with implemented wrappers (ORCA, NWChem, Gaussian09, XTB
and MOPAC) that are available from your *PATH* environment variable. To check the expected high and low level methods are
available:

.. code-block:: python

  >>> from autode import methods
  >>> methods.get_hmethod()
  <autode.wrappers.ORCA.ORCA object at XXXXXXXXXXX>
  >>> methods.get_lmethod()
  <autode.wrappers.XTB.XTB object object at XXXXXXXXXXX>


If a MethodUnavailable exception is raised see the :doc:`troubleshooting page <troubleshooting>`.

Quick test
----------

If the high and/or low level methods have been installed for the first time
it may be useful to check they're installed correctly. To run a quick optimisation
of H\ :sub:`2`\:

.. code-block:: python

  >>> from autode import methods, Molecule
  >>> h2 = Molecule(smiles='[H][H]')
  >>> h2.optimise(method=methods.get_lmethod())
  >>> h2.optimise(method=methods.get_hmethod())
  >>> h2.energy
  -1.164013897947
  >>> h2.atoms
  [[H, 0.3805, 0.0000, 0.0000], [H, -0.3805, 0.0000, 0.0000]]

If an AtomsNotFound exception is raised it is likely that the electronic structure
package is not correctly installed.
