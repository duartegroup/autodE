Configuration
=============

The *Config* can be modified from the input script for full customization of
the calculations. By default low level optimisations are performed at PBE-D3BJ/def2-SVP,
optimisations at PBE0-D3BJ/def2-SVP and single points at PBE0-D3BJ/def2-TZVP in
ORCA if it is available.

For example, to use Gaussian09 as the high level electronic structure method

.. code-block:: python

  >>> Config.hcode = 'g09'

To set the number of cores available and the memory per core (in MB)

.. code-block:: python

  >>> Config.n_cores = 8
  >>> Config.max_core = 4000

Calculation parameters also can be changed, e.g to use B3LYP/def2-TZVP single point
energies in ORCA

.. code-block:: python

  >>> from autode.wrappers.keywords import SinglePointKeywords
  >>> Config.ORCA.keywords.sp = SinglePointKeywords(['SP', 'B3LYP', 'def2-TZVP'])

To add diffuse functions with the ma scheme to the def2-SVP default optimisation
basis set for optimisations

.. code-block:: python

  >>> from autode.wrappers.keywords import OptKeywords, HessianKeywords
  >>> Config.ORCA.keywords.opt = OptKeywords(['Opt', 'PBE0', 'D3BJ', 'ma-def2-SVP'])
  >>> Config.ORCA.keywords.hess = HessianKeywords(['Freq', 'PBE0', 'D3BJ', 'ma-def2-SVP'])
  >>> Config.ORCA.keywords.opt_ts = OptKeywords(['OptTS', 'Freq', 'PBE0', 'D3BJ', 'ma-def2-SVP'])


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

