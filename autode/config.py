class Config:
    # -----------------------------------------------
    # Total number of cores available
    #
    n_cores = 8
    #
    # -----------------------------------------------
    # Per core memory available in MB
    #
    max_core = 4000
    #
    # -----------------------------------------------

    class ORCA:
        # Parameters for ORCA
        path = None

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

    class XTB:
        path = None

    class MOPAC:
        path = None
        licence_path = None

    class PSI4:
        path = None
        scan_keywords = None
        conf_opt_keywords = None
        opt_keywords = None
        opt_ts_keywords = None
        hess_keywords = None
        opt_ts_block = None
        sp_keywords = None
