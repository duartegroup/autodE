.. _adding_est:

******************
Adding QM Wrappers
******************

|

Electronic structure (QM) calculations are handled in **autodE** by writing input
files and parsing the output through a set of wrapper classes in
(`autode.wrappers <https://github.com/duartegroup/autodE/tree/master/autode/wrappers>`_).
To add a new wrapper for e.g. TurboMole:

0. Make a new file (*autode/wrappers/turbomole.py*) containing a :class:`TurboMole` class.

1. Subclass :class:`ElectronicStructureMethod` and implement all abstract methods.

2. Add a :class:`TurboMole` class within :class:`ade.Config` see `here <https://github.com/duartegroup/autodE/blob/master/autode/config.py>`_, with default keywords for each implemented type of calculation.

3. Optional: Include keywords in keyword wrappers e.g. *wrappers/functionals.py*, so keywords may have associated citations.

4. Optional: Add implemented implicit solvent names to *autode/solvent/solvents.py*.

5. Write tests, following the examples in *tests/test_orca_calc.py*.
