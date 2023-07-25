Configuration
=============

Configuration is handled with :code:`ade.Config` and can be modified for full
customization of the calculations. By default, high-level optimisations are
performed at PBE0-D3BJ/def2-SVP and single points at PBE0-D3BJ/def2-TZVP.


Calculations
------------

General
*******

The high-level electronic structure code defaults to the first available
from {ORCA, Gaussian09, Gaussian16, NWChem, QChem} and the low-level from
{XTB, MOPAC}. To select Gaussian09 as the high-level method:

.. code-block:: python

  >>> import autode as ade
  >>> ade.Config.hcode = 'g09'

Similarly, with the low-level:

.. code-block:: python

  >>> ade.Config.lcode = 'MOPAC'


To set the number of cores available and the memory per core (in MB), to use a maximum
of 32 GB for the whole calculation:

.. code-block:: python

  >>> ade.Config.n_cores = 8
  >>> ade.Config.max_core = 4000

------------

Keywords
********

**autodE** uses wrappers around common keywords used in QM calculations to allow
easy setting of e.g. a DFT functional.


.. code-block:: python

  >>> kwds = ade.Config.ORCA.keywords.sp
  >>> kwds.functional
  Functional(pbe0)


To modify the functional for single point energies, in ORCA:


.. code-block:: python

  >>> kwds.functional = 'B3LYP'


Alternatively, reassign to a whole new set of keywords:


.. code-block:: python

  >>> ade.Config.ORCA.keywords.sp = ['SP', 'B3LYP', 'def2-TZVP']


To add diffuse functions with the *ma* scheme to the def2-SVP default
basis set for optimisations:

.. code-block:: python

  >>> ade.Config.ORCA.keywords.set_opt_basis_set('ma-def2-SVP')


.. note::
    `set_opt_basis_set` sets the basis set in keywords.grad, keywords.opt_ts
    keywords.opt, keywords.low_opt and keywords.hess.

------------

Temporary configuration
**********************

It is also possible to change configuration temporarily, by using the context
manager:

.. code-block:: python

    >>> ade.Config.ORCA.keywords.opt.functional
    Functional(pbe0)
    >>> ade.Config.n_cores = 4
    >>> mol = ade.Molecule(smiles='CCO')
    >>> with ade.temporary_config():
    >>>     ade.Config.n_cores = 9
    >>>     ade.Config.ORCA.keywords.opt.funcitonal = 'B3LYP'
    >>>     # this calculation will run with 9 cores and B3LYP functional
    >>>     mol.optimise(method=ade.methods.ORCA())
    >>> # when context manager returns previous state of Config is restored
    >>> ade.Config.n_cores
    4
    >>> ade.Config.ORCA.keywords.opt.functional
    Functional(pbe0)

When the context manager exits, the previous state of the configuration is
restored.

.. warning::
    Note that the context manager works by saving the state of the Config
    when it is called and restoring the state when it exits. The way Python
    handles object references means that any references taken before or inside
    the context manager will become useless after it exits. Please see the example
    below for details.

.. code-block:: python

    >>> kwds = ade.Config.ORCA.keywords  # kwds refers to an object inside Config.ORCA
    >>> with temporary_config():
    ...     kwds.opt.functional = 'B3LYP'
    ...     mol.optimise(method=ade.method.ORCA())
    ...     # this works successfully
    >>> # when context manager exits, all variables in Config are restored, including Config.ORCA
    >>> # But kwds still refers to an object from old Config.ORCA
    >>> kwds.opt.functional
    Functional(B3LYP)
    >>> ade.Config.ORCA.opt.functional  # current config
    Functional(pbe0)

As seen from the above example, the variable :code:`kwds` is useless once the
context manager exits, and changes to :code:`kwds` no longer affects autodE. It is
best to always modify :code:`Config` directly.

------------

XTB as a hmethod
****************

To use XTB as the *hmethod* for minima and TS optimisations within Gaussian use the `xtb-gaussian <https://github.com/aspuru-guzik-group/xtb-gaussian>`_ wrapper
and some default options. Note that the string to call `xtb-gaussian` will need to be modified with the appropriate keywords for spin and solvent, e.g., "xtb-gaussian --alpb water".

.. code-block:: python

  >>> kwds = ade.Config.G16.keywords
  >>> kwds.sp = ["External='xtb-gaussian'", "IOp(3/5=30)"]
  >>> kwds.low_opt = ["External='xtb-gaussian'", "Opt(Loose, NoMicro)", "IOp(3/5=30)"]
  >>> kwds.opt = ["External='xtb-gaussian'", "Opt(NoMicro)", "IOp(3/5=30)"]
  >>> kwds.opt_ts = ["External='xtb-gaussian'", "Opt(TS, CalcFC, NoEigenTest, MaxCycles=100, MaxStep=10, NoTrustUpdate, NoMicro)", "IOp(3/5=30)"]
  >>> kwds.hess = ["External='xtb-gaussian'", "Freq", "Geom(Redundant)", "IOp(3/5=30)"]
  >>> kwds.grad = ["External='xtb-gaussian'", 'Force(NoStep)', "IOp(3/5=30)"]

To use XTB within ORCA copy the :code:`xtb` binary to the folder where the :code:`orca` binary is located and rename it :code:`otool_xtb`, then
set the keywords to use. For example

.. code-block:: python

  >>> kwds = ade.Config.ORCA.keywords
  >>> kwds.sp = ['SP', 'PBE0', 'def2-SVP']
  >>> kwds.opt = ['Opt', 'XTB2']
  >>> kwds.low_opt = ['Opt', 'XTB2']
  >>> kwds.hess = ['NumFreq', 'XTB2']
  >>> kwds.grad = ['EnGrad', 'XTB2']
  >>> kwds.opt_ts = ['OptTS', 'NumFreq', 'XTB2\n',
    '%geom\n'
    'NumHess true\n'
    'Calc_Hess true\n'
    'Recalc_Hess 30\n'
    'Trust -0.1\n'
    'MaxIter 150\n'
    'end']


------------

Other
*****

See the `config file <https://github.com/duartegroup/autodE/blob/master/autode/config.py>`_
to see all the options.

.. note::
    NWChem currently only supports solvents for DFT, other methods must not have
    a solvent.

------------

Logging
-------

To set the logging level to one of {DEBUG, INFO, WARNING, ERROR} set the :code:`AUTODE_LOG_LEVEL`
environment variable, in bash::

    $ export AUTODE_LOG_LEVEL=INFO

To output the log to a file set e.g. *autode.log*::

    $ export AUTODE_LOG_FILE=autode.log

To log with timestamps and colours::

    $ conda install coloredlogs


To set the logging level permanently add the above export statements to
your *bash_profile*.

In case of Windows command prompt, use the set command to set environment
variables::

    > set AUTODE_LOG_LEVEL=INFO

For powershell, use :code:`$env`::

    > $env:AUTODE_LOG_FILE = 'INFO'
