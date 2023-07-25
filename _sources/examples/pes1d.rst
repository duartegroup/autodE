*****************
1D PES Generation
*****************


**autodE** allows for both potential energy surface (PES) to be constructed
where other degrees of freedom are frozen (unrelaxed) or allowed to
optimise (relaxed).


Unrelaxed
---------

For the O-H dissociation curve in H\ :sub:`2`\ O at the XTB level:

.. literalinclude:: ../common/OH_PES_unrelaxed.py

Out (OH_PES_unrelaxed.png):

.. image:: ../common/OH_PES_unrelaxed.png
   :width: 550
   :align: center

For the same O-H 1D PES scan using a selection of different DFT methods:

.. literalinclude:: ../common/OH_PES_unrelaxed_DFT.py

Out (OH_PES_unrelaxed2.png):

.. image:: ../common/OH_PES_unrelaxed_DFT.png
   :width: 550
   :align: center

Relaxed
-------

A relaxed 1D PES can be generated and plotted using the default :code:`plot`
method for the same O-H stretch using:

.. literalinclude:: ../common/OH_PES_relaxed.py

Out (OH_PES_relaxed.png):

.. image:: ../common/OH_PES_relaxed.png
   :width: 500
   :align: center

.. code-block::

    r_1 (Å)   E (Ha)
    0.6500   -4.93638
    0.7464   -5.01741
    0.8429   -5.05749
    0.9393   -5.07023
    1.0357   -5.06677
    1.1321   -5.05464
    1.2286   -5.03825
    1.3250   -5.02002
    1.4214   -5.00116
    1.5179   -4.98253
    1.6143   -4.96472
    1.7107   -4.94807
    1.8071   -4.93276
    1.9036   -4.91887
    2.0000   -4.90642
