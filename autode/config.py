from autode.wrappers.keywords import Keywords


class Config:
    # ----------------------------------------------------------------------------------------------
    # Total number of cores available
    #
    n_cores = 4
    #
    # ----------------------------------------------------------------------------------------------
    # Per core memory available in MB
    #
    max_core = 4000
    #
    # ----------------------------------------------------------------------------------------------
    # DFT code to use. If set to None then the highest priority available code will be used:
    # 1. 'orca', 2. 'g09' 3. 'nwchem'
    #
    hcode = None
    #
    # ----------------------------------------------------------------------------------------------
    # Semi-empirical/tight binding method to use. If set to None then the highest priority available
    # will be used:   1. 'xtb', 2. 'mopac'
    #
    lcode = None
    #
    # ----------------------------------------------------------------------------------------------
    # When a transition state is found save it it /path_to_autode_install/transition_states/lib/
    # if ts_template_folder_path is not set to override it
    #
    make_ts_template = True
    #
    ts_template_folder_path = None
    #
    # ----------------------------------------------------------------------------------------------
    # Save plots with dpi = 1000
    high_quality_plots = True
    #
    # ----------------------------------------------------------------------------------------------
    # RMSD in angstroms threshold for conformers. Larger values will remove more conformers that
    # need to be calculated but also reduces the chance that the lowest energy conformer is found
    #
    rmsd_threshold = 0.3
    #
    # ----------------------------------------------------------------------------------------------
    # Total number of conformers generated in find_lowest_energy_conformer() for single molecules/TSs
    #
    num_conformers = 300
    # ----------------------------------------------------------------------------------------------
    # Maximum random displacement in angstroms for conformational searching
    #
    max_atom_displacement = 4.0
    # ----------------------------------------------------------------------------------------------
    # Number of evenly spaced points on a sphere that will be used to generate NCI and Reactant/
    # Product complex conformers. Total number of conformers will be:
    # num_complex_sphere_points × num_complex_random_rotations × (n molecules in complex - 1)
    #
    num_complex_sphere_points = 10
    # ----------------------------------------------------------------------------------------------
    # Number of random rotations of a molecule that is added to a NCI or Reactant/Product complex;
    # larger numbers will be slower, but more likely to find the minimum
    #
    num_complex_random_rotations = 10
    # ----------------------------------------------------------------------------------------------

    class ORCA:
        # ------------------------------------------------------------------------------------------
        # Parameters for orca                        https://sites.google.com/site/orcainputlibrary/
        # ------------------------------------------------------------------------------------------
        #
        # Note: path can be unset and will be assigned if it can be found in $PATH
        path = None

        keywords = Keywords(low_opt=['LooseOpt', 'PBE', 'RI', 'D3BJ', 'def2-SVP', 'def2/J'],
                            grad=['EnGrad', 'PBE', 'RI', 'D3BJ', 'def2-SVP', 'def2/J'],
                            opt=['Opt', 'PBE0', 'RIJCOSX', 'D3BJ', 'def2-SVP', 'def2/J'],
                            opt_ts=['OptTS', 'Freq', 'PBE0', 'RIJCOSX', 'D3BJ', 'def2-SVP', 'def2/J'],
                            hess=['Freq', 'PBE0', 'RIJCOSX', 'D3BJ', 'def2-SVP', 'def2/J'],
                            optts_block='%geom\nCalc_Hess true\nRecalc_Hess 30\nTrust -0.1\nMaxIter 150\nend',
                            sp=['SP', 'PBE0', 'RIJCOSX', 'D3BJ', 'def2/J', 'def2-TZVP'])

        # Implicit solvent in ORCA is either treated with CPCM or SMD, the former has support for a
        # VdW surface construction which provides better geometry convergence (https://doi.org/10.1002/jcc.26139)
        # SMD is in general more accurate, but does not (yet) have support for the VdW charge scheme
        # 1. 'cpcm', 2. 'smd'
        solvation_type = 'cpcm'

    class G09:
        # ------------------------------------------------------------------------------------------
        # Parameters for g09                                      https://gaussian.com/glossary/g09/
        # ------------------------------------------------------------------------------------------
        #
        # Note: path can be unset and will be assigned if it can be found in $PATH
        path = None
        #
        keywords = Keywords(low_opt=['PBEPBE/Def2SVP', 'Opt=Loose'],
                            grad=['PBEPBE/Def2SVP', 'Force(NoStep)'],
                            opt=['PBE1PBE/Def2SVP', 'Opt'],
                            opt_ts=['PBE1PBE/Def2SVP', 'Freq',
                                    'Opt=(TS, CalcFC, NoEigenTest, MaxCycles=100, MaxStep=10, NoTrustUpdate)'],
                            hess=['PBE1PBE/Def2SVP', 'Freq'],
                            sp=['PBE1PBE/Def2TZVP'])

    class NWChem:
        # ------------------------------------------------------------------------------------------
        # Parameters for nwchem                         http://www.nwchem-sw.org/index.php/Main_Page
        # ------------------------------------------------------------------------------------------
        #
        # Note: path can be unset and will be assigned if it can be found in $PATH
        path = None
        #
        svp_basis_block = 'basis\n  *   library Def2-SVP\nend',
        tzvp_basis_block = 'basis\n  *   library Def2-TZVP\nend'
        #
        keywords = Keywords(low_opt=['driver\n'
                                     '  gmax 0.002\n'
                                     '  grms 0.0005\n'
                                     '  xmax 0.01\n'
                                     '  xrms 0.007\n'
                                     '  eprec 0.00003\n'
                                     'end',
                                     svp_basis_block,
                                     'dft\n'
                                     '  maxiter 100\n'
                                     '  xc xpbe96 cpbe96\n'
                                     'end',
                                     'task dft optimize'],
                            grad=['basis\n'
                                  '  *   library Def2-SVP\n'
                                  'end',
                                  'dft\n'
                                  '  xc xpbe96 cpbe96\n'
                                  'end',
                                  'task dft gradient'],
                            opt=['driver\n'
                                 '  gmax 0.0003\n'
                                 '  grms 0.0001\n'
                                 '  xmax 0.004\n'
                                 '  xrms 0.002\n'
                                 '  eprec 0.000005\n'
                                 'end',
                                 svp_basis_block,
                                 'dft\n'
                                 '  maxiter 100\n'
                                 '  xc pbe0\n'
                                 'end',
                                 'task dft optimize',
                                 'task dft property'],
                            opt_ts=['driver\n'
                                    '  maxiter 100\n'
                                    '  gmax 0.0003\n'
                                    '  grms 0.0001\n'
                                    '  xmax 0.004\n'
                                    '  xrms 0.002\n'
                                    '  eprec 0.000005\n'
                                    'end',
                                    svp_basis_block,
                                    'dft\n'
                                    '  xc pbe0\n'
                                    'end',
                                    'task dft saddle',
                                    'task dft freq',
                                    'task dft property'],
                            hess=[svp_basis_block, 'dft\n  xc pbe0\nend', 'task dft freq'],
                            sp=[tzvp_basis_block, 'dft\n  xc pbe0\nend', 'task dft energy'])

    class XTB:
        # ------------------------------------------------------------------------------------------
        # Parameters for xtb    https://www.chemie.uni-bonn.de/pctc/mulliken-center/software/xtb/xtb
        # ------------------------------------------------------------------------------------------
        #
        # Note: path can be unset and will be assigned if it can be found in $PATH
        path = None
        #
        keywords = Keywords()

    class MOPAC:
        # ------------------------------------------------------------------------------------------
        # Parameters for mopac                                                  http://openmopac.net
        # ------------------------------------------------------------------------------------------
        #
        # Note: path can be unset and will be assigned if it can be found in $PATH
        path = None
        #
        # Note: all optimisations at this low level will be in the gas phase using the keywords_list
        # specified here. Solvent in mopac is defined by EPS and the dielectric
        keywords = Keywords(low_opt=['PM7', 'PRECISE'])
