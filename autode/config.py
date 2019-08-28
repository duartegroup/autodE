class Config(object):
    #
    # Total number of cores available
    #
    n_cores = 8
    #
    orca_max_core = 4000
    #
    # Parameters for ORCA
    scan_keywords = ['LooseOpt', 'PBE', 'RI', 'D3BJ', 'def2-SVP', 'def2/J']
    conf_opt_keywords = ['LooseOpt', 'PBE', 'RI', 'D3BJ', 'def2-SVP', 'def2/J']
    opt_keywords = ['Opt', 'PBE0', 'RIJCOSX', 'D3BJ', 'def2-SVP', 'def2/J']
    opt_ts_keywords = ['OptTS', 'Freq', 'PBE0', 'RIJCOSX', 'D3BJ', 'def2-SVP', 'def2/J']
    hess_keywords = ['Freq', 'PBE0', 'RIJCOSX', 'D3BJ', 'def2-SVP', 'def2/J']
    opt_ts_block = ('%geom\nCalc_Hess true\n'
                    'Recalc_Hess 40\n'
                    'Trust 0.2\n'
                    'MaxIter 100\n'
                    'end')
    sp_keywords = ['SP', 'PBE0', 'RIJCOSX', 'D3BJ', 'def2/J', 'def2-TZVP']
    #
    # Paths:
    # The full path to an executable needs to be provided for any electronic structure method invoked
    #
    path_to_orca = '/usr/local/orca_4_1_1/orca'
    path_to_xtb = '/usr/local/xtb/bin/xtb'
    path_to_mopac = None
    path_to_mopac_licence = None
