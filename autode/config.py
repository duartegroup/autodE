class Config:
    # ----------------------------------------------------------------------------------------------
    # Total number of cores available
    #
    n_cores = 8
    #
    # ----------------------------------------------------------------------------------------------
    # Per core memory available in MB
    #
    max_core = 4000
    #
    # ----------------------------------------------------------------------------------------------
    # DFT code to use. If set to None then the highest priority available code will be used:
    #                                                                      1. 'ORCA', 2. 'PSI4'
    #
    hcode = 'ORCA'
    #
    # Semi-empirical/tight binding method to use. If set to None then the highest priority available
    # will be used:   1. 'XTB', 2. 'MOPAC'
    #
    lcode = 'XTB'
    #
    # ----------------------------------------------------------------------------------------------

    class ORCA:
        # ------------------------------------------------------------------------------------------
        # Parameters for ORCA                        https://sites.google.com/site/orcainputlibrary/
        # ------------------------------------------------------------------------------------------
        #
        # Note: path can be unset and will be assigned if it can be found in $PATH
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
        # ------------------------------------------------------------------------------------------
        # Parameters for XTB    https://www.chemie.uni-bonn.de/pctc/mulliken-center/software/xtb/xtb
        # ------------------------------------------------------------------------------------------
        #
        # Note: path can be unset and will be assigned if it can be found in $PATH
        path = None

    class MOPAC:
        # ------------------------------------------------------------------------------------------
        # Parameters for MOPAC                                                  http://openmopac.net
        # ------------------------------------------------------------------------------------------
        #
        # Note: path can be unset and will be assigned if it can be found in $PATH
        path = None
        licence_path = None
        #
        # Note: all optimisations at this low level will be in the gas phase using the keywords
        # specified here. Solvent in MOPAC is defined by EPS and the dielectric
        keywords = ['PM7', 'PRECISE']

    class PSI4:
        # ------------------------------------------------------------------------------------------
        # Parameters for PSI4                                                 http://www.psicode.org
        # ------------------------------------------------------------------------------------------
        #
        # Note: path can be unset and will be assigned if it can be found in $PATH
        path = None
        scan_keywords = ['set {basis def2-svp\n'
                         'g_convergence gau_loose'
                         '}',
                         "optimize('scf', dft_functional = 'PBE0-D3BJ')"]
        conf_opt_keywords = ['set {basis def2-svp\ng_convergence gau_loose}',
                             "optimize('scf', dft_functional = 'PBE0-D3BJ')"]
        opt_keywords = ['set basis def2-svp',
                        "optimize('scf', dft_functional = 'PBE0-D3BJ')"]
        opt_ts_keywords = ['set {full_hess_every 30\n'
                           'opt_type ts\n'
                           'basis def2-svp\n}',
                           "optimize('scf', dft_functional = 'PBE0-D3BJ', dertype=1)",
                           "frequencies('scf', dft_functional = 'PBE0-D3BJ', dertype=1)"]
        hess_keywords = None
        opt_ts_block = 'set intrafrag_step_limit 0.1'
        sp_keywords = ['set basis def2-tzvp',
                       "energy('scf', dft_functional = 'PBE0-D3BJ')"]
