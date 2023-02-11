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
