# distutils: language = c++
from libcpp.vector cimport vector
from libcpp cimport bool as bool_t

cdef extern from "molecule.h" namespace "autode":
    cdef cppclass Molecule:
        # Constructors
        Molecule()
        Molecule(vector[double])

        # Attributes exposed to Python
        int n_atoms
        double energy
        vector[double] coords
        Dihedrals _dihedrals

        void rotate_dihedrals()


cdef extern from "potentials.h" namespace "autode":
    cdef cppclass Potential:
        pass


cdef extern from "potentials.h" namespace "autode":
    cdef cppclass RBPotential(Potential):
        RBPotential()
        RBPotential(int,             # Exponent
                    vector[bool_t],  # Bonds
                    vector[double],  # Ideal pairwise distances (r0)
                    vector[double],  # Force constants (k)
                    vector[double])  # Pairwise repulsion coefficients


cdef extern from "potentials.h" namespace "autode":
    cdef cppclass RDihedralPotential(Potential):
        RDihedralPotential()
        RDihedralPotential(int,             # Exponent
                           vector[bool_t])  # Pairs that should be considered


cdef extern from "optimisers.h" namespace "autode":
    cdef cppclass SDOptimiser:
        SDOptimiser()

        # Method
        void run(Potential &,
                 Molecule &,
                 int,        # Max number of iterations
                 double,     # Energy difference
                 double)     # Initial step size


cdef extern from "optimisers.h" namespace "autode":
    cdef cppclass SDDihedralOptimiser(SDOptimiser):
        SDDihedralOptimiser()




cdef extern from "dihedrals.h" namespace "autode":
    cdef cppclass Dihedral:
        Dihedral()
        Dihedral(double,           # Angle
                 vector[int],      # Axis
                 vector[bool_t],   # Rotation indexes
                 int)              # Origin


cdef extern from "dihedrals.h" namespace "autode":
    cdef cppclass Dihedrals:
        Dihedrals()
        Dihedrals(vector[Dihedral])

        void push_back(Dihedral &)
        void rotate(Molecule &)
