# distutils: language = c++
from libcpp.vector cimport vector
from libcpp cimport bool as bool_t


cdef extern from "include/molecule.h" namespace "autode":
    cdef cppclass Molecule:
        # Constructors
        Molecule()
        Molecule(vector[double])

        # Attributes exposed to Python
        int n_atoms
        double energy
        vector[double] coords
        vector[Dihedral] _dihedrals

        void rotate_dihedrals()


cdef extern from "include/potentials.h" namespace "autode":
    cdef cppclass Potential:
        pass


cdef extern from "include/potentials.h" namespace "autode":
    cdef cppclass RBPotential(Potential):
        RBPotential()
        RBPotential(int,             # Exponent
                    vector[bool_t],  # Bonds
                    vector[double],  # Ideal pairwise distances (r0)
                    vector[double],  # Force constants (k)
                    vector[double])  # Pairwise repulsion coefficients


cdef extern from "include/potentials.h" namespace "autode":
    cdef cppclass RDihedralPotential(Potential):
        RDihedralPotential()
        RDihedralPotential(int,             # Exponent
                           vector[bool_t])  # Pairs that should be considered


cdef extern from "include/potentials.h" namespace "autode":
    cdef cppclass RRingDihedralPotential(Potential):
        RRingDihedralPotential()
        RRingDihedralPotential(int,            # Exponent
                               vector[bool_t], # Pairs that should be considered
                               vector[int],    # Pair of atom indexes to close
                               double)         # Distance for ring closure


cdef extern from "include/optimisers.h" namespace "autode":
    cdef cppclass SDOptimiser:
        SDOptimiser()

        # Method
        void run(Potential &,
                 Molecule &,
                 int,        # Max number of iterations
                 double,     # Energy difference
                 double)     # Initial step size


cdef extern from "include/optimisers.h" namespace "autode":
    cdef cppclass SDDihedralOptimiser(SDOptimiser):
        SDDihedralOptimiser()


cdef extern from "include/optimisers.h" namespace "autode":
    cdef cppclass GridDihedralOptimiser:
        GridDihedralOptimiser()

        void run(Potential &,
                 Molecule &,
                 int,        # Maximum number of grid points
                 double,     # Energy difference for final SD
                 double)  # Initial step size for SD minimisation


cdef extern from "include/optimisers.h" namespace "autode":
    cdef cppclass SGlobalDihedralOptimiser:
        SGlobalDihedralOptimiser()

        void run(Potential &,
                 Molecule &,
                 int,        # Number of maximum total steps
                 double,     # Final tolerance on SD minimisation
                 double)     # Initial step size for SD minimisation

cdef extern from "include/dihedrals.h" namespace "autode":
    cdef cppclass Dihedral:
        Dihedral()
        Dihedral(double,           # Angle
                 vector[int],      # Axis
                 vector[bool_t],   # Rotation indexes
                 int)              # Origin

cdef extern from "include/idpp.h" namespace "autode":
    cdef cppclass IdppParams:
        double k_spr
        bool_t sequential
        bool_t debug
        double rmsgtol
        int maxiter
        double add_img_maxgtol
        double add_img_maxiter

cdef extern from "include/idpp.h" namespace "autode":
    void calculate_idpp_path(double* init_coords_ptr,
                             double* final_coords_ptr,
                             int coords_len,
                             int n_images,
                             double* all_coords_ptr,
                             const IdppParams & params) except +

cdef extern from "include/idpp.h" namespace "autode":
    double get_path_length(double* init_coords_ptr,
                           double* final_coords_ptr,
                           int coords_len,
                           int n_images,
                           const IdppParams & params) except +

cdef extern from "include/idpp.h" namespace "autode":
    void relax_path(double* all_coords_ptr,
                    int coords_len,
                    int n_images,
                    const IdppParams & params) except +
