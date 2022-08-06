***************
Thermochemistry
***************

Thermochemical contributions are calculated in **autodE** using the ideal gas
model and variants thereof. The molar enthalpy is,

.. math::
    H = E_\text{elec} + E_\text{internal} + E_\text{ZPE} + RT

and Gibbs free energy,

.. math::
    G = H - TS

where :math:`T` is the temperature and :math:`R` the gas constant. See
`here <https://github.com/duartegroup/autodE/tree/master/autode/common/thermochemistry.pdf>`_
for a more in-depth mathematical background. With a completed electronic
structure calculation free energies can be obtained using either RRHO,
shifted-RRHO (from Truhlar[1]) and qRRHO (from Grimme[2]). Global parameters
can also be set in :code:`autode.Config`, which apply to all thermochemical
calculations including those when calculating a profile with
:code:`rxn.calculate_reaction_profile(free_energy=True)`

******

General
-------

To calculate thermochemical contributions (enthalpy, Gibbs free energy) in **autodE** the minimal required input
is

.. code-block:: Python

    import autode as ade

    water = ade.Molecule(smiles='O')
    water.calc_thermo()
    print(f'E = {water.energy:.6f}',
          f'H = {water.enthalpy:.6f}',
          f'G = {water.free_energy:.6f}',
          f'units = {water.energy.units}')

**Out**:
:code:`E = -76.269260
H = -76.245955
G = -76.264398
units = Unit(Ha)`


For a non-default set method and keywords:

.. code-block:: Python

    import autode as ade

    water = ade.Molecule(smiles='O')
    water.calc_thermo(method=ade.methods.G09(),
                      keywords=['PBEPBE', 'def2SVP', 'Freq'])


ORCA
----
To calculate thermochemical contributions from a completed ORCA Hessian file (*H2O_hess_orca.hess*):

.. code-block:: Python

    import autode as ade

    mol = ade.Molecule('H2O_hess_orca.xyz')
    orca = ade.methods.ORCA()

    calc = ade.Calculation(name='H2O',
                           molecule=mol,
                           method=orca,
                           keywords=orca.keywords.hess)
    calc.output.filename = 'H2O_hess_orca.hess'

    mol.calc_thermo(calc=calc, temp=298.15, ss='1atm', sn=1)
    print(mol.g_cont)

**Out**: :code:`0.00145779587867773`

which differs from the ORCA-calculated value (0.00145673 Ha) by <0.001 kcal mol\ :sup:`-1`\. To
calculate a total free energy including :math:`E_\text{elec}` both a .out and .hess file need to be present with the
same basename. For example:

.. code-block:: Python

    import autode as ade

    mol = ade.Molecule('H2O_hess_orca.xyz')
    orca = ade.methods.ORCA()

    calc = ade.Calculation(name='H2O',
                           molecule=mol,
                           method=orca,
                           keywords=orca.keywords.hess)
    calc.output.filename = 'H2O_hess_orca.out'

    mol.calc_thermo(calc=calc)
    print(f'H = {mol.enthalpy:.6f} Ha\n'
          f'G = {mol.free_energy:.6f} Ha')

**Out**:

.. code-block::

    H = -76.249086 Ha
    G = -76.267526 Ha

where, without any arguments to :code:`calc_thermo`, the default method uses room temperature (298.15 K),
a one molar (1 M) standard state (appropriate for molecules in solution), Grimme's qRRHO treatment of
low frequency vibrational modes and a calculated symmetry number, which in this case is two (C\ :sub:`2v` \ symmetry).

******

Gaussian
--------

Likewise from a Gaussian output file of butane (*butane_hess_g09.log*):

.. code-block:: Python

    import autode as ade

    mol = ade.Molecule('butane.xyz')
    g09 = ade.methods.G09()

    calc = ade.Calculation(name='butane',
                           molecule=mol,
                           method=g09,
                           keywords=g09.keywords.hess)
    calc.output.filename = 'butane_hess_g09.log'

    mol.calc_thermo(calc=calc, temp=298.15, ss='1atm', sn=1, lfm_method='igm')
    print(mol.g_cont)

**Out**: :code:`0.10419152589407932`

which differs from the Gaussian-calculated value (0.104216 Ha) by ~0.01 kcal mol\ :sup:`-1`\.

.. note::

    Gaussian 09 has very tight tolerances on symmetry and uses a pure
    harmonic oscillator treatment of low frequency modes.

*****************

Frequency scaling
-----------------

Default methods use frequency scaling automatically to generate the most accurate
thermochemistry possible. Unscaled frequencies can be obtained by setting

.. code-block:: Python

    ade.Config.freq_scale_factor = 1.0

.. note::

    The above examples were generated using unscaled frequencies.


References
----------

[1] R. F. Ribeiro, A. V. Marenich, C. J. Cramer and D. G. Truhlar, *Phys. Chem. B* 2011, **115**, 14556.

[2] S. Grimme, *Chem. Eur. J.* 2012, **18**, 9955.
