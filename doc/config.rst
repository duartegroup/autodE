Configuration
=============

The Config file can be modified from the input script for full customization of
the calculations. By default low level optimisations are performed at PBE-D3BJ/def2-SVP,
optimisations at PBE0-D3BJ/def2-SVP and single points at PBE0-D3BJ/def2-TZVP.

For example, to use Gaussian09 and XTB as the high and low level electronic structure methods

.. code-block:: python

  >>> Config.hcode = 'g09'
  >>> Config.lcode = 'xtb'

To set the number of cores available and the memory per core (in MB)

.. code-block:: python

  >>> Config.n_cores = 8
  >>> Config.max_core = 4000

Further, the parameters used in the calculations can be changed, e.g to change
how the single point energies are calculated

.. code-block:: python

  >>> Config.ORCA.keywords.sp = ['SP', 'B3LYP', 'def2-TZVP']

To add diffuse functions with the ma scheme to the def2-SVP default optimisation
basis set for optimisations

.. code-block:: python

  >>> Config.ORCA.keywords.opt = ['Opt', 'PBE0', 'D3BJ', 'ma-def2-SVP']
  >>> Config.ORCA.keywords.hess = ['Freq', 'PBE0', 'D3BJ', 'ma-def2-SVP']
  >>> Config.ORCA.keywords.opt_ts = ['OptTS', 'Freq', 'PBE0', 'D3BJ', 'ma-def2-SVP']


See the `config file <https://github.com/duartegroup/autodE/blob/master/autode/config.py>`_
to see all the options.

.. note::
    NWChem currently only supports solvents for DFT, other methods must not have
a solvent.

Logging
-------

To set the logging level to one of {INFO, WARNING, ERROR} set the AUTODE_LOG_LEVEL
environment variable, in bash::

    $ export AUTODE_LOG_LEVEL=INFO

