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
    pass


class SolventNotFound(Exception):
    pass


class SolventsDontMatch(Exception):
    pass


class RDKitFailed(Exception):
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


class CouldNotGetProperty(Exception):

    def __init__(self, message):
        super().__init__(message)
