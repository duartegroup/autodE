import os

if os.getenv('AUTODE_FIXUNIQUE') != 'False':
    raise RuntimeError('Please set $AUTODE_FIXUNIQUE to False to run the '
                       'tests\n e.g. export AUTODE_FIXUNIQUE=False')
