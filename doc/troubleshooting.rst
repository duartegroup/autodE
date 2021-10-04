Troubleshooting
===============

------------


Conda Solve Fails
-----------------

If conda cannot solve the environment after attempting to install the dependencies create a new
`virtual environment <https://docs.conda.io/projects/conda/en/latest/user-guide/concepts/environments.html>`_ with::

   $ conda create -n autode_env
   $ conda activate autode_env

then install. The **autodE** environment will need to be activated each time a new shell is opened, with:
``conda activate autode_env``.


------------

MethodUnavailable
-----------------

If high and/or low level electronic structure methods cannot be found in your *PATH*
environment variable it is possible to set the paths manually in :code:`ade.Config` for e.g. XTB:

.. code-block:: python

    >>> from autode import methods, Config
    >>> methods.get_lmethod()
    autode.exceptions.MethodUnavailable
    >>> Config.XTB.path = '/path/to/xtb/bin/xtb'
    >>> methods.get_lmethod()
    XTB(available = True)


alternatively, add to the PATH environment variable e.g. in bash::

    $ export PATH=/path/to/xtb/bin/:$PATH


to set this permanently, add the above line to ~/.bash_profile or ~/.bashrc on Linux.


------------


A Transition State Cannot be Located
-------------------------------------
Automatically finding transition states (TS) with **autodE** can sometimes fail due to either
a misidentification of the TS as incorrect or an insufficiently good TS guess geometry being found.
If a TS hasn't been found first check the *reaction_name_path.png* visualisation of the initial path
and proceed depending on whether a peak is or is not present.

Without a path peak
*******************
Oh no! without an initial peak a TS won't be able to be found. Perhaps the reaction is electronically
barrierless at this level of theory. If there are anions consider adding diffuse functions, if they are not
already present e.g.:

.. code-block:: python

    >>> from autode import Config
    >>> Config.ORCA.keywords.set_opt_basis_set('ma-def2-SVP')

With a path peak
****************

If there is a peak then either (1) the TS guess geometry is not close enough to the TS for successful
optimisation. To reduce the step size:

.. code-block:: python

    >>> from autode import Config
    >>> Config.min_step_size = 0.02

Otherwise (2), it may be that the TS optimiser cannot locate the saddle point because of Hessian drift. Consider
reducing the number of intermediate steps between Hessian updates (i.e. `Config.ORCA.keywords.opt_ts` for ORCA).

If you've found an interesting case where a TS cannot be found please do get in touch, we're always on the lookout
for examples to improve the method!
