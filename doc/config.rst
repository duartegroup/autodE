Config
======

The Config file can be modified from the input script for full customization of the calculations.

To set which Electronic Structure Methods to use

.. code-block:: python

  Config.hcode = 'ORCA'
  Config.lcode = 'XTB'

To set the number of cores available and the memory per core (in MB)

.. code-block:: python

  Config.n_cores = 8
  Config.max_core = 4000

Further, the parameters used in the calculations can be changed, e.g to change how the single point energies are calculated

.. code-block:: python

  Config.ORCA.sp_keywords = ['SP', 'PBE0', 'RIJCOSX', 'D3BJ', 'def2/J', 'def2-TZVP']

We suggest you look in the config file to see the format used.
Note: NWChem currently only supports DFT.

