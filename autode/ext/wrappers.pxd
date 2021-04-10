# distutils: language = c++
from libcpp.vector cimport vector
from libcpp cimport bool as bool_t

cdef extern from "molecule.h" namespace "autode":
    cdef cppclass Molecule:
        # Constructors
        Molecule()
        Molecule(vector[double], vector[bool_t])

        # Attributes exposed to Python
        int n_atoms
        double energy
        vector[double] coords


cdef extern from "potentials.h" namespace "autode":
    cdef cppclass Potential:
        pass


cdef extern from "potentials.h" namespace "autode":
    cdef cppclass RBPotential(Potential):
        RBPotential()
        RBPotential(int,             # Exponent
                    vector[double],  # Ideal pairwise distances (r0)
                    vector[double],  # Force constants (k)
                    vector[double])  # Pairwise repulsion coefficients


cdef extern from "optimisers.h" namespace "autode":
    cdef cppclass SDOptimiser:
        SDOptimiser()

        # Method
        void run(Potential &,
                 Molecule &,
                 int,        # Max number of iterations
                 double,     # Energy difference
                 double)     # Initial step size
