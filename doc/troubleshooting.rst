Troubleshooting
===============

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

