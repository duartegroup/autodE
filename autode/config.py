from autode.wrappers.keywords import KeywordsSet


class Config:
    # -------------------------------------------------------------------------
    # Total number of cores available
    #
    n_cores = 4
    #
    # -------------------------------------------------------------------------
    # Per core memory available in MB
    #
    max_core = 4000
    #
    # -------------------------------------------------------------------------
    # DFT code to use. If set to None then the highest priority available code
    # will be used:
    # 1. 'orca', 2. 'g09' 3. 'nwchem'
    #
    hcode = None
    #
    # -------------------------------------------------------------------------
    # Semi-empirical/tight binding method to use. If set to None then the
    # highest priority available will be used:   1. 'xtb', 2. 'mopac'
    #
    lcode = None
    #
    # -------------------------------------------------------------------------
    # When using explicit solvent is stable this will be uncommented
    #
    # explicit_solvent = False
    #
    # -------------------------------------------------------------------------
    # Setting to keep input files, otherwise they will be removed
    #
    keep_input_files = True
    #
    # -------------------------------------------------------------------------
    # By default templates are saved to /path/to/autode/transition_states/lib/
    # unless ts_template_folder_path is set
    #
    ts_template_folder_path = None
    #
    # Whether or not to create and save transition state templates
    make_ts_template = True
    # -------------------------------------------------------------------------
    # Save plots with dpi = 400
    high_quality_plots = True
    #
    # -------------------------------------------------------------------------
    # RMSD in angstroms threshold for conformers. Larger values will remove
    # more conformers that need to be calculated but also reduces the chance
    # that the lowest energy conformer is found
    #
    rmsd_threshold = 0.3
    #
    # -------------------------------------------------------------------------
    # Total number of conformers generated in find_lowest_energy_conformer()
    # for single molecules/TSs
    #
    num_conformers = 300
    # -------------------------------------------------------------------------
    # Maximum random displacement in angstroms for conformational searching
    #
    max_atom_displacement = 4.0
    # -------------------------------------------------------------------------
    # Number of evenly spaced points on a sphere that will be used to generate
    # NCI and Reactant and Product complex conformers. Total number of
    # conformers will be:
    #   (num_complex_sphere_points ×
    #              num_complex_random_rotations) ^ (n molecules in complex - 1)
    #
    num_complex_sphere_points = 10
    # -------------------------------------------------------------------------
    # Number of random rotations of a molecule that is added to a NCI or
    # Reactant/Product complex
    #
    num_complex_random_rotations = 10
    # -------------------------------------------------------------------------
    # For more than 2 molecules in a complex the conformational space explodes,
    # so limit the maximum number to this value
    #
    max_num_complex_conformers = 300
    # -------------------------------------------------------------------------
    # Use the high + low level method to find the lowest energy
    # conformer, to use energies at the low_opt level of the low level code
    # set this to False
    #
    hmethod_conformers = True
    # -------------------------------------------------------------------------

    class ORCA:
        # ---------------------------------------------------------------------
        # Parameters for orca   https://sites.google.com/site/orcainputlibrary/
        # ---------------------------------------------------------------------
        #
        # Path can be unset and will be assigned if it can be found in $PATH
        path = None

        keywords = KeywordsSet(low_opt=['LooseOpt', 'PBE',  'D3BJ', 'def2-SVP'],
                               grad=['EnGrad', 'PBE0', 'D3BJ', 'def2-SVP'],
                               opt=['Opt', 'PBE0', 'RIJCOSX', 'D3BJ', 'def2-SVP', 'def2/J'],
                               opt_ts=['OptTS', 'Freq', 'PBE0', 'RIJCOSX', 'D3BJ', 'def2-SVP', 'def2/J'],
                               hess=['Freq', 'PBE0', 'RIJCOSX', 'D3BJ', 'def2-SVP', 'def2/J'],
                               sp=['SP', 'PBE0', 'RIJCOSX', 'D3BJ', 'def2/J', 'def2-TZVP'],
                               optts_block=('%geom\n'
                                            'Calc_Hess true\n' 
                                            'Recalc_Hess 30\n'
                                            'Trust -0.1\n'
                                            'MaxIter 150\n'
                                            'end'))

        # Implicit solvent in ORCA is either treated with CPCM or SMD, the
        # former has support for a VdW surface construction which provides
        # better geometry convergence (https://doi.org/10.1002/jcc.26139) SMD
        # is in general more accurate, but does not (yet) have support for the
        # VdW charge scheme. Use either 1. 'cpcm', 2. 'smd'
        implicit_solvation_type = 'cpcm'

    class G09:
        # ---------------------------------------------------------------------
        # Parameters for g09                 https://gaussian.com/glossary/g09/
        # ---------------------------------------------------------------------
        #
        # path can be unset and will be assigned if it can be found in $PATH
        path = None
        #
        disp = 'EmpiricalDispersion=GD3BJ'
        grid = 'integral=ultrafinegrid'

        keywords = KeywordsSet(low_opt=['PBEPBE/Def2SVP', 'Opt=Loose', disp, grid],
                               grad=['PBE1PBE/Def2SVP', 'Force(NoStep)', disp, grid],
                               opt=['PBE1PBE/Def2SVP', 'Opt', disp, grid],
                               opt_ts=['PBE1PBE/Def2SVP', 'Freq', disp, grid,
                                       'Opt=(TS, CalcFC, NoEigenTest, '
                                       'MaxCycles=100, MaxStep=10, '
                                       'NoTrustUpdate)'],
                               hess=['PBE1PBE/Def2SVP', 'Freq', disp, grid],
                               sp=['PBE1PBE/Def2TZVP', disp, grid])

        # Only SMD implemented
        implicit_solvation_type = 'smd'

    class G16:
        # ---------------------------------------------------------------------
        # Parameters for g16                   https://gaussian.com/gaussian16/
        # ---------------------------------------------------------------------
        #
        # path can be unset and will be assigned if it can be found in $PATH
        path = None
        #
        disp = 'EmpiricalDispersion=GD3BJ'
        keywords = KeywordsSet(low_opt=['PBEPBE/Def2SVP', 'Opt=Loose', disp],
                               grad=['PBE1PBE/Def2SVP', 'Force(NoStep)', disp],
                               opt=['PBE1PBE/Def2SVP', 'Opt', disp],
                               opt_ts=['PBE1PBE/Def2SVP', 'Freq', disp,
                                       'Opt=(TS, CalcFC, NoEigenTest, '
                                       'MaxCycles=100, MaxStep=10, '
                                       'NoTrustUpdate)'],
                               hess=['PBE1PBE/Def2SVP', 'Freq', disp],
                               sp=['PBE1PBE/Def2TZVP', disp])

        # Only SMD implemented
        implicit_solvation_type = 'smd'

    class NWChem:
        # ---------------------------------------------------------------------
        # Parameters for nwchem    http://www.nwchem-sw.org/index.php/Main_Page
        # ---------------------------------------------------------------------
        #
        # Path can be unset and will be assigned if it can be found in $PATH
        path = None
        #
        svp_basis_block = 'basis\n  *   library Def2-SVP\nend'
        tzvp_basis_block = 'basis\n  *   library Def2-TZVP\nend'
        #
        # Note that the default NWChem level is PBE0 and PBE rather than
        # PBE0-D3BJ and PBE-D3BJ as only D3 is available
        keywords = KeywordsSet(low_opt=['driver\n'
                                        '  gmax 0.002\n'
                                        '  grms 0.0005\n'
                                        '  xmax 0.01\n'
                                        '  xrms 0.007\n'
                                        '  eprec 0.00003\n'
                                        '  maxiter 50\n'
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
                                     '  xc pbe0\n'
                                     'end',
                                     'task dft gradient'],
                               opt=['driver\n'
                                    '  gmax 0.0003\n'
                                    '  grms 0.0001\n'
                                    '  xmax 0.004\n'
                                    '  xrms 0.002\n'
                                    '  eprec 0.000005\n'
                                    '  maxiter 100\n'
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
                               hess=[svp_basis_block, 'dft\n  xc pbe0\nend',
                                     'task dft freq'],
                               sp=[tzvp_basis_block, 'dft\n  xc pbe0\nend',
                                   'task dft energy'])

        # Only SMD implemented
        implicit_solvation_type = 'smd'

    class XTB:
        # ---------------------------------------------------------------------
        # Parameters for xtb                  https://github.com/grimme-lab/xtb
        # ---------------------------------------------------------------------
        #
        # path can be unset and will be assigned if it can be found in $PATH
        path = None
        #
        keywords = KeywordsSet()
        #
        # Only GBSA implemented
        implicit_solvation_type = 'gbsa'

    class MOPAC:
        # ---------------------------------------------------------------------
        # Parameters for mopac                             http://openmopac.net
        # ---------------------------------------------------------------------
        #
        # path can be unset and will be assigned if it can be found in $PATH
        path = None
        #
        # Note: all optimisations at this low level will be in the gas phase
        # using the keywords_list specified here. Solvent in mopac is defined
        # by EPS and the dielectric
        keywords = KeywordsSet(low_opt=['PM7', 'PRECISE'])
        #
        # Only COSMO implemented
        implicit_solvation_type = 'cosmo'

