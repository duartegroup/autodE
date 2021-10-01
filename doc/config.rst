Configuration
=============

Configuration is handled with :code:`ade.Config` and can be modified for full
customization of the calculations. By default, high level optimisations are
performed at PBE0-D3BJ/def2-SVP and single points at PBE0-D3BJ/def2-TZVP. To
edit the default configuration permanently edit the config file, the location
of which can be accessed with:

.. code-block:: python

    >>> import autode as ade
    >>> ade.config.location
    '/a/path/.../lib/python3.X/site-packages/autode.../autode/config.py'


------------

Calculations
------------

General
*******

The high-level electronic structure code defaults to using the first available
from {ORCA, Gaussian09, Gaussian16, NWChem} and the low-level from {XTB, MOPAC}.
To select Gaussian09 as the high-level method:

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
  >>> ade.Config.ORCA.keywords.sp = ade.SinglePointKeywords(['SP', 'B3LYP', 'def2-TZVP'])


To add diffuse functions with the *ma* scheme to the def2-SVP default
basis set for optimisations

.. code-block:: python

  >>> ade.Config.ORCA.keywords.set_opt_basis_set('ma-def2-SVP')

.. note::
    `set_opt_basis_set` sets the basis set in keywords.grad, keywords.opt_ts
    keywords.opt, keywords.low_opt and keywords.hess.

------------

XTB as a hmethod
****************

To use XTB as the *hmethod* for minima and TS optimisations with the `xtb-gaussian <https://github.com/aspuru-guzik-group/xtb-gaussian>`_ wrapper
and some default options

.. code-block:: python

  >>> ade.Config.G16.keywords.sp = SinglePointKeywords([f"external='xtb-gaussian'"])
  >>> ade.Config.G16.keywords.low_opt = OptKeywords([f"external='xtb-gaussian'", "opt=loose"])
  >>> ade.Config.G16.keywords.opt = OptKeywords([f"external='xtb-gaussian'", "opt"])
  >>> ade.Config.G16.keywords.opt_ts = OptKeywords([f"external='xtb-gaussian'", 'Opt=(TS, CalcFC, NoEigenTest, MaxCycles=100, MaxStep=10, NoTrustUpdate)', "freq"])
  >>> ade.Config.G16.keywords.hess = HessianKeywords([f"external='xtb-gaussian'", 'freq'])
  >>> ade.Config.G16.keywords.grad = GradientKeywords([f"external='xtb-gaussian'", 'Force(NoStep)'])

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

To set the logging level to one of {INFO, WARNING, ERROR} set the AUTODE_LOG_LEVEL
environment variable, in bash::

    $ export AUTODE_LOG_LEVEL=INFO

To output the log to a file set e.g. *autode.log*::

    $ export AUTODE_LOG_FILE=autode.log

To log with timestamps and colours::

    $ conda install coloredlogs


