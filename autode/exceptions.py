class XYZsNotFound(Exception):
    pass


class NoAtomsInMolecule(Exception):
    pass


class NoAtomsToPrint(Exception):
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


class RDKitFailed(Exception):
    pass


class BondsInSMILESAndGraphDontMatch(Exception):
    pass


class XYZfileDidNotExist(Exception):
    pass


class XYZfileWrongFormat(Exception):
    pass
