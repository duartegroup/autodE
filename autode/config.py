import os
from typing import Any
from autode.values import Frequency, Distance, Allocation
from autode.wrappers.keywords import implicit_solvent_types as solv
from autode.wrappers.keywords import KeywordsSet, MaxOptCycles
from autode.wrappers.keywords.basis_sets import (
    def2svp,
    def2tzvp,
    def2ecp,
    def2tzecp,
)
from autode.wrappers.keywords.functionals import pbe0
from autode.wrappers.keywords.dispersion import d3bj
from autode.wrappers.keywords.ri import rijcosx

location = os.path.abspath(__file__)


class _ConfigClass:
    # -------------------------------------------------------------------------
    # Total number of cores available
    #
    n_cores = 4
    # -------------------------------------------------------------------------
    # Per core memory available
    #
    max_core = Allocation(4, units="GB")
    # -------------------------------------------------------------------------
    # DFT code to use. If set to None then the highest priority available code
    # will be used:
    # 1. 'orca', 2. 'g09' 3. 'nwchem'
    #
    hcode = None
    # -------------------------------------------------------------------------
    # Semi-empirical/tight binding method to use. If set to None then the
    # highest priority available will be used:   1. 'xtb', 2. 'mopac'
    #
    lcode = None
    # -------------------------------------------------------------------------
    # When using explicit solvent is stable this will be uncommented
    #
    # explicit_solvent = False
    #
    # -------------------------------------------------------------------------
    # Setting to keep input files, otherwise they will be removed
    #
    keep_input_files = True
    # -------------------------------------------------------------------------
    # Use a different base directory for calculations with low-level methods
    # e.g. /dev/shm with a low level method, if None then will use the default
    # in tempfile.mkdtemp
    #
    ll_tmp_dir = None
    # -------------------------------------------------------------------------
    # By default templates are saved to /path/to/autode/transition_states/lib/
    # unless ts_template_folder_path is set
    #
    ts_template_folder_path = None
    # -------------------------------------------------------------------------
    # Whether or not to create and save transition state templates
    #
    make_ts_template = True
    # -------------------------------------------------------------------------
    # Save plots with dpi = 400
    #
    high_quality_plots = True
    # -------------------------------------------------------------------------
    # RMSD in angstroms threshold for conformers. Larger values will remove
    # more conformers that need to be calculated but also reduces the chance
    # that the lowest energy conformer is found
    #
    rmsd_threshold = Distance(0.3, units="Å")
    # -------------------------------------------------------------------------
    # Total number of conformers generated in find_lowest_energy_conformer()
    # for single molecules/TSs
    #
    num_conformers = 300
    # -------------------------------------------------------------------------
    # Maximum random displacement in angstroms for conformational searching
    #
    max_atom_displacement = Distance(4.0, units="Å")
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
    # Minimum and maximum step size to use for the adaptive path search
    #
    min_step_size = Distance(0.05, units="Å")
    max_step_size = Distance(0.3, units="Å")
    # -------------------------------------------------------------------------
    # Heuristic for pruning the bond rearrangement set. If there are only bond
    # rearrangements that involve small rings then TSs involving small rings
    # are possible. However, when there are multiple possibilities involving
    # the same set of atoms then discard any rearrangements that would involve
    # a 3 or 4-membered TS e.g. skip the possible 4-membered TS for a Cope
    # rearrangement in hexadiene
    #
    skip_small_ring_tss = True
    # -------------------------------------------------------------------------
    # Minimum magnitude of the imaginary frequency (cm-1) to consider for a
    # 'true' TS. For very shallow saddle points this may need to be reduced
    # to e.g. -10 cm-1. Although most TSs have |v_imag| > 100 cm-1 this
    # threshold is designed to be conservative
    #
    min_imag_freq = Frequency(-40, units="cm-1")
    # -------------------------------------------------------------------------
    # Configuration parameters for ideal gas free energy calculations. Can be
    # configured to use different standard states, quasi-rigid rotor harmonic
    # oscillator (qRRHO) or pure RRHO
    #
    #  One of: '1M', '1atm'
    standard_state = "1M"
    #
    # Method to treat low frequency modes (LFMs). Either standard RRHO ('igm'),
    # Truhlar's method where all frequencies below a threshold are scaled to
    # a shifted value (see J. Phys. Chem. B, 2011, 115, 14556), Grimme's
    # method of interpolating between HO and RR (i.e. qRRHO, see
    # Chem. Eur. J. 2012, 18, 9955), or 'minenkov' where free rotor/vibrational
    # interpolation is useed for U and S (i.e. mRRHO, see
    # J. Comput. Chem., 2023 44, 1807)
    #
    # One of: 'igm', 'truhlar', 'grimme', 'minenkov'
    lfm_method = "grimme"
    #
    # Parameters for Grimme's method (only used when lfm_method='grimme'),
    # w0 is a frequency in cm-1
    grimme_w0 = Frequency(100, units="cm-1")
    grimme_alpha = 4
    #
    # Parameters for Truhlar's method (only used when lfm_method='truhlar')
    # vibrational frequencies below this value (cm-1) will be shifted to this
    # value before the entropy is calculated
    vib_freq_shift = Frequency(100, units="cm-1")
    # -------------------------------------------------------------------------
    # Frequency scale factor, useful for DFT functions known to have a
    # systematic error. This value must be between 0 and 1 inclusive. For
    # example, PBEh-3c has a scale factor of 0.95.
    #
    freq_scale_factor = None
    # -------------------------------------------------------------------------
    # Minimum number of atoms that are removed for truncation to be used in
    # locating TSs. Below this number any truncation is skipped
    #
    min_num_atom_removed_in_truncation = 10
    # -------------------------------------------------------------------------
    # Flag for allowing free energies to be calculated with association
    # complexes. This is *not* recommended to be turned on due to the
    # approximations made in the entropy calculations.
    #
    allow_association_complex_G = False
    # -------------------------------------------------------------------------
    # Flag to allow use of an experimental timeout function wrapper for
    # Windows, using loky. The default case has no timeout for Windows, and
    # timeout only works on Linux/macOS. This flag is ignored on Linux/macOS.
    #
    use_experimental_timeout = False
    # -------------------------------------------------------------------------

    class ORCA:
        # ---------------------------------------------------------------------
        # Parameters for orca   https://sites.google.com/site/orcainputlibrary/
        # ---------------------------------------------------------------------
        #
        # Path can be unset and will be assigned if it can be found in $PATH
        path = None
        #
        # File extensions to copy when a calculation completes
        copied_output_exts = [".out", ".hess", ".xyz", ".inp", ".pc"]

        optts_block = (
            "\n%geom\n"
            "Calc_Hess true\n"
            "Recalc_Hess 20\n"
            "Trust -0.1\n"
            "MaxIter 100\n"
            "end"
        )

        keywords = KeywordsSet(
            low_opt=[
                "LooseOpt",
                pbe0,
                rijcosx,
                d3bj,
                def2svp,
                "def2/J",
                MaxOptCycles(10),
            ],
            grad=["EnGrad", pbe0, rijcosx, d3bj, def2svp, "def2/J"],
            low_sp=["SP", pbe0, rijcosx, d3bj, def2svp, "def2/J"],
            opt=["Opt", pbe0, rijcosx, d3bj, def2svp, "def2/J"],
            opt_ts=[
                "OptTS",
                "Freq",
                pbe0,
                rijcosx,
                d3bj,
                def2svp,
                "def2/J",
                optts_block,
            ],
            hess=["Freq", pbe0, rijcosx, d3bj, def2svp, "def2/J"],
            sp=["SP", pbe0, rijcosx, d3bj, def2tzvp, "def2/J"],
            ecp=def2ecp,
        )

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
        grid = "integral=ultrafinegrid"
        optts_block = (
            "Opt=(TS, CalcFC, NoEigenTest, MaxCycles=100, "
            "MaxStep=10, NoTrustUpdate)"
        )

        keywords = KeywordsSet(
            low_opt=[pbe0, def2svp, "Opt=Loose", MaxOptCycles(10), d3bj, grid],
            grad=[pbe0, def2svp, "Force(NoStep)", d3bj, grid],
            low_sp=[pbe0, def2svp, d3bj, grid],
            opt=[pbe0, def2svp, "Opt", d3bj, grid],
            opt_ts=[pbe0, def2svp, "Freq", d3bj, grid, optts_block],
            hess=[pbe0, def2svp, "Freq", d3bj, grid],
            sp=[pbe0, def2tzvp, d3bj, grid],
            ecp=def2tzecp,
        )

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
        ts_str = (
            "Opt=(TS, CalcFC, NoEigenTest, MaxCycles=100, MaxStep=10, "
            "NoTrustUpdate, RecalcFC=30)"
        )

        keywords = KeywordsSet(
            low_opt=[pbe0, def2svp, "Opt=Loose", d3bj, MaxOptCycles(10)],
            grad=[pbe0, def2svp, "Force(NoStep)", d3bj],
            low_sp=[pbe0, def2svp, d3bj],
            opt=[pbe0, def2svp, "Opt", d3bj],
            opt_ts=[pbe0, def2svp, "Freq", d3bj, ts_str],
            hess=[pbe0, def2svp, "Freq", d3bj],
            sp=[pbe0, def2tzvp, d3bj],
            ecp=def2tzecp,
        )

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
        # PBE0-D3BJ and PBE-D3BJ as only D3 is available. The optimisation
        # keywords contain 'gradient' as the optimisation is driven by autodE
        keywords = KeywordsSet(
            low_opt=[def2svp, pbe0, MaxOptCycles(10), "task dft gradient"],
            grad=[def2svp, pbe0, "task dft gradient"],
            low_sp=[def2svp, pbe0, "task dft energy"],
            opt=[def2svp, pbe0, MaxOptCycles(100), "task dft gradient"],
            opt_ts=[def2svp, pbe0, MaxOptCycles(50), "task dft gradient"],
            hess=[def2svp, pbe0, "task dft freq"],
            sp=[def2tzvp, pbe0, "task dft energy"],
            ecp=def2ecp,
        )

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
        #
        # Force constant used for harmonic restraints in constrained
        # optimisations (Ha/a0)
        force_constant = 2
        #
        # Electronic temperature for all calculations (Kelvin)
        # None means unset (default), set to 300.0 to have 300K for example
        electronic_temp = None
        #
        # Version of xTB hamiltonian parameterisation: 0,1 or 2
        # corresponding to GFN0-xTB, GFN1-xTB, GFN2-xTB respectively
        # When unset, uses the default
        gfn_version = None

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
        keywords = KeywordsSet(low_opt=["PM7", "PRECISE"])
        #
        # Only COSMO implemented
        implicit_solvation_type = solv.cosmo

    class QChem:
        # ---------------------------------------------------------------------
        # Parameters for QChem                          https://www.q-chem.com/
        # ---------------------------------------------------------------------
        #
        # path can be unset and will be assigned if it can be found in $PATH
        path = None
        #
        # Default set of keywords to use for different types of calculation
        keywords = KeywordsSet(
            low_opt=[pbe0, def2svp, "jobtype opt", MaxOptCycles(10), d3bj],
            grad=[pbe0, def2svp, "jobtype force", d3bj],
            low_sp=[pbe0, def2svp, d3bj],
            opt=[pbe0, def2svp, "jobtype opt", d3bj],
            opt_ts=[pbe0, def2svp, "jobtype TS", d3bj],
            hess=[pbe0, def2svp, "jobtype Freq", d3bj],
            sp=[pbe0, def2tzvp, d3bj],
            ecp=def2ecp,
        )

        #
        # Only SMD is implemented
        implicit_solvation_type = solv.smd

    # =========================================================================
    # =============               End                        ==================
    # =========================================================================

    def __setattr__(self, key, value):
        """Custom setters"""

        if not hasattr(self, key):
            raise KeyError(f"Cannot set {key}. Not present in ade.Config")

        if key == "max_core":
            value = Allocation(value).to("MB")

        if key == "freq_scale_factor":
            if value is not None:
                if not (0.0 < value <= 1.0):
                    raise ValueError(
                        "Cannot set the frequency scale factor "
                        "outside of (0, 1]"
                    )

                value = float(value)

        if key in ("max_atom_displacement", "min_step_size", "max_step_size"):
            if float(value) < 0:
                raise ValueError(f"Distances cannot be negative. Had: {value}")

            value = Distance(value).to("ang")

        return super().__setattr__(key, value)


def _instantiate_config_opts(cls: type) -> Any:
    """
    Instantiate a config class containing options defined
    as class variables. It generates an instance of the
    class, and then creates instance variables of the same
    name as class variables, recursively converting any
    nested class into instances.
    (This is required because class variables are not pickled,
    only instance variables are)

    Args:
        cls (type): Must be a class containing class
         variables (not instance)

    Returns:
        (Any): The generated class instance
    """
    if not isinstance(cls, type):
        raise ValueError("Must be a class, not an instance")
    cls_instance = cls()
    for name, attr in cls.__dict__.items():
        if name.startswith("__"):
            continue
        if isinstance(attr, type):
            attr_val = _instantiate_config_opts(attr)  # recursive
        else:
            attr_val = attr
        setattr(cls_instance, name, attr_val)
    return cls_instance


# Single instance of the configuration
Config = _instantiate_config_opts(_ConfigClass)
