import os

# For ORCA/Gaussian etc. calculations to be skipped there needs to be no
# attempt to make calculation names unique if they have a different input, so
# set the appropriate flag
os.environ['AUTODE_FIXUNIQUE'] = 'False'
