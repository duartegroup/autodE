**********************************
Non-covalent Interaction Complexes
**********************************

**autodE** allows for the systematic search of a NCI complexes conformational
space. For example, to find the lowest energy structure of the water trimer

.. literalinclude:: ../common/water_trimer.py

Out (visualised)

.. image:: ../common/water_trimer.png

|
The parameters (:code:`num_complex_sphere_points` and :code:`num_complex_random_rotations`)
define the number of generated conformers, up to :code:`ade.Config.max_num_complex_conformers`.

.. image:: ../common/water_trimer_expl.png
   :width: 550
   :align: center
