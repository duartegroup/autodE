Troubleshooting
===============

Conda Solve Fails
-----------------

If conda cannot solve the environment after attempting to install the dependencies create a new
`virtual environment <https://docs.conda.io/projects/conda/en/latest/user-guide/concepts/environments.html>`_ with::

   $ conda create -n autode_env
   $ conda activate autode_env

then install. The autodE environment will need to be activated each time a new shell is opened, with:
``conda activate autode_env``.



MethodUnavailable
-----------------

If high and/or low level electronic structure methods cannot be found in $PATH it is possible to set the paths manually
in Config for e.g. XTB:

.. code-block:: python

    >>> from autode import methods, Config
    >>> methods.get_lmethod()
    autode.exceptions.MethodUnavailable
    >>> Config.XTB.path='/path/to/xtb/bin/xtb'
    >>> methods.get_lmethod()
    <autode.wrappers.XTB.XTB object at XXXXXXXXXX>


alternatively, add to the PATH environment variable e.g. in bash::

    $ export PATH=/path/to/xtb/bin/:$PATH


to set this permanently, add the above line to ~/.bash_profile on MacOS or ~/.bashrc on Linux.

