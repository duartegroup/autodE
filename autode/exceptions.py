class AtomsNotFound(Exception):
    pass


class NoAtomsInMolecule(Exception):
    pass


class NoConformers(Exception):
    pass


class NoInputError(Exception):
    pass


class MethodUnavailable(Exception):
    pass


class SolventUnavailable(Exception):
    pass


class UnbalancedReaction(Exception):
    pass


class UnsuppportedCalculationInput(Exception):
    def __init__(self, message='Parameters not supported'):
        super().__init__(message)


class SolventNotFound(Exception):
    pass


class SolventsDontMatch(Exception):
    pass


class BondsInSMILESAndGraphDontMatch(Exception):
    pass


class XYZfileDidNotExist(Exception):
    pass


class XYZfileWrongFormat(Exception):
    pass


class ReactionFormationFalied(Exception):
    pass


class NoClosestSpecies(Exception):
    pass


class NoNormalModesFound(Exception):
    pass


class NoMolecularGraph(Exception):
    pass


class NoCalculationOutput(Exception):
    pass


class RDKitFailed(Exception):
    pass


class InvalidSmilesString(Exception):
    pass


class CouldNotGetProperty(Exception):

    def __init__(self, message):
        super().__init__(message)


class CannotSplitAcrossBond(Exception):
    """Exception for when a molecule cannot be partitioned by deleting one bond"""


class CouldNotPlotSmoothProfile(Exception):
    """Exception for when smooth plotting fails"""
