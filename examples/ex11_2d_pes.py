import autode as ade

orca = ade.methods.ORCA()
if not orca.available:
    exit('This example requires a ORCA install')

# 2+ dimensional PESs can also be calculated. For example, considering the
# identity reaction CH3 + CH4 -> CH4 + CH3
reactive_complex = ade.Molecule('CH3_CH4.xyz', mult=2)

# Create then calculate the PES
pes = ade.pes.RelaxedPESnD(reactive_complex,
                           rs={(0, 1): (3.0, 8),    # Current -> 3.0 Å in 10
                               (5, 1): (1.1, 8)})   # Current -> 1.1 Å in 10
pes.calculate(method=orca,
              keywords=['LooseOpt', 'PBE', 'def2-SV(P)'],  # Fast DFT
              n_cores=10)                        # Using 10 processing cores

# and plot the 2D surface
pes.plot(filename='CH3_CH4.pdf')

# To plot the surface with interpolation rerun this script with:
# pes.plot(filename='CH3_CH4.pdf',
#          interp_factor=3)
# NOTE: the calculations will be skipped
