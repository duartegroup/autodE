Install
=======

Dependencies
------------

- `Python <https://www.python.org/>`_ > v. 3.5
- `ORCA <https://sites.google.com/site/orcainputlibrary/home/>`_ > v. 4.1
- `XTB <https://www.chemie.uni-bonn.de/pctc/mulliken-center/software/xtb/xtb/>`_ > v. 6.1

The Python dependencies are listed in requirements.txt best satisfied using a conda install (Miniconda or Anaconda) i.e.

.. code-block::

  conda config --append channels conda-forge
  conda create -n autode_env --file requirements.txt
  conda activate autode_env

Installation
------------

Once the dependencies are satisfied, to install autodE enter the autodE folder then:

.. code-block::

  python setup.py install
