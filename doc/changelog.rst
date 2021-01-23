Changelog
=========

1.0.0b3
--------
------------

This version brings several major changes and in some instances breaks
backwards compatibility, but does feature significant improvements in speed
and accuracy for finding transition states.

Usability improvements
**********************

- :code:`species.get_distance(i, j)` is now :code:`species.distance(i, j)`

- :code:`species.set_atoms(new_atoms)`  is now properly handled with a setter so :code:`species.atoms = new_atoms` will set the new atoms

- :code:`species.n_atoms` is more robust

- :code:`species.get_coordinates()` is now :code:`species.coordinates`, returning a numpy array copy of the species coordinates (Nx3 in Å)

- :code:`species.centre()` will translate a species so it's coordinate centroid lies at the origin


Functionality improvements
**************************

- 1D and 2D potential energy surface scans have been replaced by a climbing image (CI) nudged elastic band implementation with adaptive force constants

- (CI)-NEB relaxation on a path where only a single bond has been scanned is skipped

- Initial NEB path exploration from reactants is performed with a larger average step size and at the 'low_opt' level, providing a significant acceleration for this usually bottlenecking step.

With NEB calculations now default the final bond-breaking distance is more important.
The previous implementation made use of a 1.5 Å additional shift for uncharged reactions
and 2.5 Å for charged, this however lead to possible final C-H distances of ~3.6 Å and steps
into unphysical regions. Now use use an estimate based on the distance where the bond
is mostly broken, as below


.. image:: common/XY_bde_XTB.png

where X-Y corresponds to a molecule e.g. C-C with the appropriate hydrogens added
then the BDE curve calculated at the GFN2-XTB level of theory.

- There is now a heuristic used to skip TSs that go via small rings (3, 4-membered) if there is a >4-membered equivalent (:code:`ade.Config.skip_small_ring_tss`)


Bug Fixes
*********

- Calculations are now unique based on constraints, so NEB calculations executed in the same directory are not skipped with different bond rearrangements
