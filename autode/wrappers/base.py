from shutil import which

req_methods = ['generate_input', 'calculation_terminated_normally',  'get_energy', 'optimisation_converged',
               'optimisation_nearly_converged', 'get_imag_freqs', 'get_normal_mode_displacements', 'get_final_xyzs',
               'get_scan_values_xyzs_energies']


class ElectronicStructureMethod:

    def __init__(self, name, path, req_licence=False, path_to_licence=None, aval_solvents=None):

        self.path = path if path is not None else which(name)      # If the path is not set in config.py search in $PATH
        self.aval_solvents = aval_solvents

        if req_licence:
            self.available = True if path is not None and path_to_licence is not None else False
        else:
            self.available = True if path is not None else False
