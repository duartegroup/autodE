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
    # 1. 'ORCA', 2. 'G09'
    #
    hcode = 'ORCA'
    #
    # Semi-empirical/tight binding method to use. If set to None then the highest priority available
    # will be used:   1. 'XTB', 2. 'MOPAC'
    #
    lcode = 'XTB'
    #
    make_ts_template = True
    #
    high_qual_plots = True
    #
    # File extension to use for the images made
    #
    image_file_extension = '.png'
    #
    # Whether to do explicit solvation or not
    #
    explicit_solvation = True
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
        gradients_keywords = ['EnGrad', 'PBE', 'RI', 'D3BJ', 'def2-SVP', 'def2/J']
        sp_grad_keywords = ['EnGrad', 'PBE0', 'RIJCOSX', 'D3BJ', 'def2/J', 'def2-TZVP']
        opt_keywords = ['Opt', 'PBE0', 'RIJCOSX', 'D3BJ', 'def2-SVP', 'def2/J']
        opt_ts_keywords = ['OptTS', 'Freq', 'PBE0', 'RIJCOSX', 'D3BJ', 'def2-SVP', 'def2/J']
        hess_keywords = ['Freq', 'PBE0', 'RIJCOSX', 'D3BJ', 'def2-SVP', 'def2/J']
        opt_ts_block = ('%geom\nCalc_Hess true\n'
                        'Recalc_Hess 30\n'
                        'Trust -0.1\n'
                        'MaxIter 100')
        sp_keywords = ['SP', 'PBE0', 'RIJCOSX', 'D3BJ', 'def2/J', 'def2-TZVP']

    class G09:
        # ------------------------------------------------------------------------------------------
        # Parameters for G09                                      https://gaussian.com/glossary/g09/
        # ------------------------------------------------------------------------------------------
        #
        # Note: path can be unset and will be assigned if it can be found in $PATH
        path = None

        scan_keywords = ['PBEPBE/Def2SVP', 'Opt=Loose']
        conf_opt_keywords = ['PBEPBE/Def2SVP', 'Opt=Loose']
        opt_keywords = ['PBE1PBE/Def2SVP', 'Opt']
        opt_ts_keywords = ['PBE1PBE/Def2SVP', 'Opt=(TS, CalcFC, NoEigenTest, MaxCycles=100, MaxStep=10, NoTrustUpdate)',
                           'Freq']
        hess_keywords = ['PBE1PBE/Def2SVP', 'Freq']
        sp_keywords = ['PBE1PBE/Def2TZVP']

    class NWChem:
        # ------------------------------------------------------------------------------------------
        # Parameters for NWChem                         http://www.nwchem-sw.org/index.php/Main_Page
        # ------------------------------------------------------------------------------------------
        #
        # Note: path can be unset and will be assigned if it can be found in $PATH
        path = None
        scan_keywords = ['driver\n  gmax 0.002\n  grms 0.0005\n  xmax 0.01\n  xrms 0.007\n  eprec 0.00003\nend',
                         'basis\n  *   library Def2-SVP\nend', 'dft\n  maxiter 100\n  xc xpbe96 cpbe96\nend', 'task dft optimize']
        conf_opt_keywords = ['driver\n  gmax 0.002\n  grms 0.0005\n  xmax 0.01\n  xrms 0.007\n  eprec 0.00003\nend',
                             'basis\n  *   library Def2-SVP\nend', 'dft\n  maxiter 100\n  xc xpbe96 cpbe96\nend', 'task dft optimize', 'task dft property']
        opt_keywords = ['driver\n  gmax 0.0003\n  grms 0.0001\n  xmax 0.004\n  xrms 0.002\n  eprec 0.000005\nend',
                        'basis\n  *   library Def2-SVP\nend', 'dft\n  maxiter 100\n  xc pbe0\nend', 'task dft optimize', 'task dft property']
        opt_ts_keywords = ['driver\n  maxiter 100\n  gmax 0.0003\n  grms 0.0001\n  xmax 0.004\n  xrms 0.002\n  eprec 0.000005\nend',
                           'basis\n  *   library Def2-SVP\nend', 'dft\n  xc pbe0\nend', 'task dft saddle', 'task dft freq']
        hess_keywords = ['basis\n  *   library Def2-SVP\nend',
                         'dft\n  xc pbe0\nend', 'task dft freq']
        sp_keywords = ['basis\n  *   library Def2-TZVP\nend',
                       'dft\n  xc pbe0\nend', 'task dft energy']

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
        path_to_licence = None
        #
        # Note: all optimisations at this low level will be in the gas phase using the keywords
        # specified here. Solvent in MOPAC is defined by EPS and the dielectric
        keywords = ['PM7', 'PRECISE']
