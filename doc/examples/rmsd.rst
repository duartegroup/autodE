****
RMSD
****

**autodE** can be used to calculate RMSD values between different molecules. For
example, a script that compares .xyz files in the current directory and
copies them to a folder (*unique_conformers*) if they are unique based on an
RMSD threshold

.. literalinclude:: ../common/rmsd.py

which can be used::

    $ python rmsd.py conformer1.xyz. conformer2.xyz -t 0.1


.. note::
   There are many other Python packages to calculate RMSD e.g.
   `rmsd <https://github.com/charnley/rmsd/>`_ and
   `spyrmsd <https://github.com/RMeli/spyrmsd/>`_ which may be better!
