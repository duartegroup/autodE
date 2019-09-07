from autode.config import Config
from autode.wrappers.base import ElectronicStructureMethod
from autode.wrappers.base import req_methods


MOPAC = ElectronicStructureMethod(name='mopac',
                                  path=Config.MOPAC.path,
                                  req_licence=True,
                                  path_to_licence=Config.MOPAC.licence_path)


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


# Bind all the required functions to the class definition
[setattr(MOPAC, method, globals()[method]) for method in req_methods]
