class CalculationException(Exception):
    pass


class AtomsNotFound(CalculationException):
    pass


class NoAtomsInMolecule(Exception):
    pass


class NoConformers(Exception):
    pass


class FitFailed(Exception):
    pass


class NoInputError(Exception):
    pass


class MethodUnavailable(CalculationException):
    pass


class SolventUnavailable(Exception):
    pass


class UnbalancedReaction(Exception):
    pass


class UnsuppportedCalculationInput(CalculationException):
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


class NoMapping(Exception):
    pass


class NoNormalModesFound(CalculationException):
    pass


class NoMolecularGraph(Exception):
    pass


class NoCalculationOutput(CalculationException):
    pass


class RDKitFailed(Exception):
    pass


class InvalidSmilesString(Exception):
    pass


class SMILESBuildFailed(Exception):
    pass


class CouldNotGetProperty(CalculationException):
    def __init__(self, name):
        super().__init__(f'Could not get {name}')


class CannotSplitAcrossBond(Exception):
    """A molecule cannot be partitioned by deleting one bond"""


class CouldNotPlotSmoothProfile(Exception):
    pass


class TemplateLoadingFailed(Exception):
    pass
