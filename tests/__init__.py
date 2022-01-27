import os
import autode as ade

# For ORCA/Gaussian etc. calculations to be skipped there needs to be no
# attempt to make calculation names unique if they have a different input, so
# set the appropriate flag
os.environ['AUTODE_FIXUNIQUE'] = 'False'

# Run all the tests on a single core
ade.Config.n_cores = 1
