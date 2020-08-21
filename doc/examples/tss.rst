*****************
Transition States
*****************


.. image:: ../common/cope.png


In addition to generating full reaction profiles directly **autodE** provides
automated access to transition states. Given a specific bond rearrangement
generating a TS from known structures of reactants and products can be achieved
for a simple Cope rearrangement with


.. literalinclude:: ../common/cope.py

Out

.. code-block:: python

  [-544.36]

where the xyz files used are:

.. literalinclude:: ../common/cope_r.xyz


.. literalinclude:: ../common/cope_p.xyz
