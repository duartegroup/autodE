*****************
2D PES Generation
*****************


2D scans in **autodE** are available over distances and are optimally parallelised
over the available number of cores. For example, to calculate and plot the 2D
relaxed surface over the C-C distances in a Diels-Alder reaction between ethene
and butadiene


.. literalinclude:: ../common/DA_2d.py


Out:

.. image:: ../common/DA_PES2d.png


where the xyz files used are:


.. literalinclude:: ../common/DA_r.xyz


.. literalinclude:: ../common/DA_p.xyz


and the plot has been generated using a 2-dimensional polynomial fit up to order 3.
