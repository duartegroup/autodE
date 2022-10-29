import autode as ade

# For more informative logs installing coloredlogs is recommended:
# conda install coloredlogs


# autodE writes logging information at the 'ERROR' level by default. To
# turn on logging export the AUTODE_LOG_LEVEL environment variable to
# one of: INFO, WARNING, ERROR

# Will not print any log
_ = ade.Molecule(smiles="N")

# To set the level to info in bash:
# export AUTODE_LOG_LEVEL=INFO
# then run this script again.

# To write the log to a file set pipe the output to a file e.g.
# python s_logging.py 2> ade.log
