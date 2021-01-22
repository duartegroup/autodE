import autode.wrappers.implicit_solvent_types as solv
from autode.wrappers.keywords import KeywordsSet
from autode.wrappers.basis_sets import def2svp, def2tzvp
from autode.wrappers.functionals import pbe, pbe0
from autode.wrappers.dispersion import d3bj
from autode.wrappers.ri import rijcosx


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
    # Use a different base directory for calculations with low-level methods
    # e.g. /dev/shm with a low level method, if None then will use the default
    # in tempfile.mkdtemp
    #
    ll_tmp_dir = None
    #
    # -------------------------------------------------------------------------
    # By default templates are saved to /path/to/autode/transition_states/lib/
    # unless ts_template_folder_path is set
    #
    ts_template_folder_path = None
    #
    # Whether or not to create and save transition state templates
    #
    make_ts_template = True
    #
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
    # Set to True to use single point energy evaluations to rank conformers and
    # select the lowest energy. Requires keywords.low_sp to be set and
    # hmethod_conformers = True
    # WARNING: This relies on the low-level geometry being accurate enough for
    # the system in question – switching this on without benchmarking may lead
    # to large errors!
    #
    hmethod_sp_conformers = False
    # -------------------------------------------------------------------------
    # Use adaptive force constant modification in NEB calculations to improve
    # sampling around the saddle point
    #
    adaptive_neb_k = True
    # -------------------------------------------------------------------------
    # Step size to use for the initial PES exploration in organic species (Å)
    #
    neb_step_size = 0.3

    # TODO - are these the right sort of thing??
    min_step_size = 0.1
    max_step_size = 0.3
    # -------------------------------------------------------------------------

    class ORCA:
        # ---------------------------------------------------------------------
        # Parameters for orca   https://sites.google.com/site/orcainputlibrary/
        # ---------------------------------------------------------------------
        #
        # Path can be unset and will be assigned if it can be found in $PATH
        path = None

        keywords = KeywordsSet(low_opt=['LooseOpt', pbe0,  d3bj, def2svp],
                               grad=['EnGrad', pbe0, rijcosx, d3bj, def2svp,
                                     'AutoAux'],
                               opt=['Opt', pbe0, rijcosx, d3bj, def2svp,
                                    'AutoAux'],
                               opt_ts=['OptTS', 'Freq', pbe0, rijcosx, d3bj,
                                       def2svp, 'AutoAux'],
                               hess=['Freq', pbe0, rijcosx, d3bj, def2svp,
                                     'AutoAux'],
                               sp=['SP', pbe0, rijcosx, d3bj, def2tzvp,
                                   'AutoAux'],
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
        # VdW charge scheme. Use either (1) solv.cpcm, (2) solv.smd
        implicit_solvation_type = solv.cpcm

    class G09:
        # ---------------------------------------------------------------------
        # Parameters for g09                 https://gaussian.com/glossary/g09/
        # ---------------------------------------------------------------------
        #
        # path can be unset and will be assigned if it can be found in $PATH
        path = None
        #
        grid = 'integral=ultrafinegrid'
        ts_str = ('Opt=(TS, CalcFC, NoEigenTest, MaxCycles=100, MaxStep=10, '
                  'NoTrustUpdate)')

        keywords = KeywordsSet(low_opt=[pbe0, def2svp, 'Opt=Loose',
                                        d3bj, grid],
                               grad=[pbe0, def2svp, 'Force(NoStep)',
                                     d3bj, grid],
                               opt=[pbe0, def2svp, 'Opt',
                                    d3bj, grid],
                               opt_ts=[pbe0, def2svp, 'Freq',
                                       d3bj, grid, ts_str],
                               hess=[pbe0, def2svp, 'Freq', d3bj, grid],
                               sp=[pbe0, def2tzvp, d3bj, grid])

        # Only SMD implemented
        implicit_solvation_type = solv.smd

    class G16:
        # ---------------------------------------------------------------------
        # Parameters for g16                   https://gaussian.com/gaussian16/
        # ---------------------------------------------------------------------
        #
        # path can be unset and will be assigned if it can be found in $PATH
        path = None
        #
        ts_str = ('Opt=(TS, CalcFC, NoEigenTest, MaxCycles=100, MaxStep=10, '
                  'NoTrustUpdate, RecalcFC=30)')

        keywords = KeywordsSet(low_opt=[pbe, def2svp, 'Opt=Loose', d3bj],
                               grad=[pbe0, def2svp, 'Force(NoStep)', d3bj],
                               opt=[pbe0, def2svp, 'Opt', d3bj],
                               opt_ts=[pbe0, def2svp, 'Freq', d3bj,
                                       ts_str],
                               hess=[pbe0, def2svp, 'Freq', d3bj],
                               sp=[pbe0, def2tzvp, d3bj])

        # Only SMD implemented
        implicit_solvation_type = solv.smd

    class NWChem:
        # ---------------------------------------------------------------------
        # Parameters for nwchem    http://www.nwchem-sw.org/index.php/Main_Page
        # ---------------------------------------------------------------------
        #
        # Path can be unset and will be assigned if it can be found in $PATH
        path = None
        #
        # Note that the default NWChem level is PBE0 and PBE rather than
        # PBE0-D3BJ and PBE-D3BJ as only D3 is available
        loose_opt_block = ('driver\n'
                           '  gmax 0.002\n'
                           '  grms 0.0005\n'
                           '  xmax 0.01\n'
                           '  xrms 0.007\n'
                           '  eprec 0.00003\n'
                           '  maxiter 50\n'
                           'end')

        opt_block = ('driver\n'
                     '  gmax 0.0003\n'
                     '  grms 0.0001\n'
                     '  xmax 0.004\n'
                     '  xrms 0.002\n'
                     '  eprec 0.000005\n'
                     '  maxiter 100\n'
                     'end')

        keywords = KeywordsSet(low_opt=[loose_opt_block, def2svp, pbe0,
                                        'task dft optimize'],
                               grad=[def2svp, pbe0,
                                     'task dft gradient'],
                               opt=[opt_block, def2svp, pbe0,
                                    'task dft optimize',
                                    'task dft property'],
                               opt_ts=[opt_block, def2svp, pbe0,
                                       'task dft saddle',
                                       'task dft freq',
                                       'task dft property'],
                               hess=[def2svp, pbe0,
                                     'task dft freq'],
                               sp=[def2tzvp, pbe0,
                                   'task dft energy'])

        # Only SMD implemented
        implicit_solvation_type = solv.smd

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
        implicit_solvation_type = solv.gbsa

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
        implicit_solvation_type = solv.cosmo

    # -------------------------------------------------------------------------
    # Use keyword naming prefixes. False to maintain backwards compatibility
    #
    keyword_prefixes = True
    #
    # -------------------------------------------------------------------------
