*********
Reactions
*********

Reactions in **autode** are :ref:`Reaction <reaction>` objects constructed from
either SMILES strings or :code:`Reactant` and :code:`Product` s. These are
elementary reactions, so the reactants should be linked to the products without
any intermediates. To initialise a reaction for: ethene + butadiene → cyclohexene:


.. code-block:: python

  >>> import autode as ade
  >>>
  >>> ethene = ade.Reactant(smiles='C=C')
  >>> butadiene = ade.Reactant(smiles='C=CC=C')
  >>> cyclohexene = ade.Product(smiles='C1=CCCCC1')
  >>>
  >>> rxn = ade.Reaction(ethene, butadiene, cyclohexene)

.. figure:: ../common/diels_alder.png

Reactions default to the gas phase and room temperature

.. code-block:: python

  >>> rxn.solvent is None
  True
  >>> rxn.temp  # in K
  298.15

Energy differences can be calculated for the overall reaction or to the
transition state (TS). If the energy of reactants and products has not been
calculated, then the energy differences are :code:`None`

.. code-block:: python

  >>> rxn.delta('E') is None    # ∆E_r
  True
  >>> rxn.delta('E‡') is None   # ∆E‡
  True

Calculating energies for the reactants and products allows for the reaction
energy difference to be calculated

.. code-block:: python

  >>> for mol in (ethene, butadiene, cyclohexene):
  ...     mol.optimise(method=ade.methods.XTB())
  >>>
  >>> rxn.delta('E').to('kcal mol-1')
  Energy(-67.44178 kcal mol-1)

If a TS has not been located for the reaction then it is assumed to be
barrierless and the barrier estimated from a diffusion limited process

.. code-block:: python

  >>> rxn.is_barrierless and rxn.delta('E‡').estimated
  True
  >>> rxn.delta('E‡').to('kcal mol-1')
  Energy(4.35491 kcal mol-1)

To optimise the reactants and products then locate the transition state using
8 cores

.. code-block:: python

  >>> ade.Config.n_cores = 8
  >>> rxn.optimise_reacs_prods()
  >>> rxn.locate_transition_state()
  >>>
  >>> rxn.ts
  TransitionState(TS_g1R2_X_ll_ad_2-3_4-5, n_atoms=16, charge=0, mult=1)
  >>> # ∆E‡ is now no longer an estimate
  >>> rxn.delta('E‡').to('kcal mol-1')
  Energy(14.30068 kcal mol-1)

