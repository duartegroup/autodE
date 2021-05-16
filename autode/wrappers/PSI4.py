from autode.config import Config
from autode.wrappers.base import ElectronicStructureMethod


class PSI4(ElectronicStructureMethod):

    # TODO: Implement abstract methods
    # TODO: Set path

    def __init__(self):

        super().__init__('psi4',
                         path=Config.PSI4.path,
                         keywords_set=Config.PSI4.keywords,
                         implicit_solvation_type=Config.PSI4.implicit_solvation_type,
                         doi='10.1002/wcms.93')
