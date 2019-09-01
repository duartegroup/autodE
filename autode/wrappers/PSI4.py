from autode.config import Config
from autode.wrappers.base import ElectronicStructureMethod
from autode.wrappers.base import req_methods


PSI4 = ElectronicStructureMethod(name='psi4',
                                 path=Config.PSI4.path,
                                 aval_solvents=None)


def generate_input(calc):
    raise NotImplementedError


def calculation_terminated_normally(calc):
    raise NotImplementedError


def get_energy(calc):
    raise NotImplementedError


def optimisation_converged(calc):
    raise NotImplementedError


def optimisation_nearly_converged(calc):
    raise NotImplementedError


def get_imag_freqs(calc):
    raise NotImplementedError


def get_normal_mode_displacements(calc, mode_number):
    raise NotImplementedError


def get_final_xyzs(calc):
    raise NotImplementedError


def get_scan_values_xyzs_energies(calc):
    raise NotImplementedError


# Bind all the required functions to the class definition
[setattr(PSI4, method, globals()[method]) for method in req_methods]
