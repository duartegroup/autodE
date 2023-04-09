class AutodeException(Exception):
    """Base autodE exception"""


# --------------------- Calculation exceptions --------------------------------
class CalculationException(AutodeException):
    """Base calculation exception when an external autodE calculation fails"""


class AtomsNotFound(CalculationException):
    """Exception for atomic coordinates/identities being found"""


class MethodUnavailable(CalculationException):
    """Exception for an autodE wrapped method not being available"""


class UnsupportedCalculationInput(CalculationException):
    """Exception for an autodE calculation input not being valid"""

    def __init__(self, message="Parameters not supported"):
        super().__init__(message)


class NoInputError(CalculationException):
    """Exception for a autodE calculation having no input e.g atoms"""


class NoCalculationOutput(CalculationException):
    """Exception for a calculation file that should exist not existing"""


class CouldNotGetProperty(CalculationException):
    """Exception for where a property e.g. energy cannot be found in a
    calculation output file"""

    def __init__(self, *args, name=None):
        if name is not None:
            super().__init__(f"Could not get {name}")

        else:
            super().__init__(*args)


class NotImplementedInMethod(CalculationException):
    """Exception for where a method is not implemented in a wrapped method"""


# -----------------------------------------------------------------------------
class NoAtomsInMolecule(AutodeException):
    """Exception for no atoms existing in a molecule"""


class NoConformers(AutodeException):
    """Exception for no conformers being present in a molecule"""


class SolventUnavailable(AutodeException):
    """Exception for a solvent not being available for a calculation"""


class UnbalancedReaction(AutodeException):
    """Exception for an autodE reaction not being balanced in e.g. N_atoms"""


class SolventNotFound(AutodeException):
    """Exception for a solvent not being in the list of available solvents"""


class SolventsDontMatch(AutodeException):
    """Exception for where the solvent in reactants is different to products"""


class BondsInSMILESAndGraphDontMatch(AutodeException):
    """Exception for a partially disjoint set of bonds in SMILES and a graph"""


class XYZfileDidNotExist(AutodeException, FileNotFoundError):
    """Exception for an xyz file not existing"""


class XYZfileWrongFormat(AutodeException):
    """Exception for an xyz being the wrong format"""


class ReactionFormationFailed(AutodeException):
    """Exception for where a reaction cannot be built. e.g. if there are no
    reactants"""


class NoMapping(AutodeException):
    """Exception for where there is no mapping/bijection between two graphs"""


class NoMolecularGraph(AutodeException):
    """Exception for a molecule not having a set graph"""


class RDKitFailed(AutodeException):
    """Exception for where RDKit fails to generate conformers"""


class InvalidSmilesString(AutodeException):
    """Exception for a SMILES string being invalid"""


class SMILESBuildFailed(AutodeException):
    """Exception for where building an example 3D structure cannot be built
    from a SMILES string"""


class FailedToSetRotationIdxs(AutodeException):
    """Exception for the atoms that need to be rotating in 3D building not
    being found"""


class FailedToAdjustAngles(AutodeException):
    """Exception for when the internal angles in a ring cannot be adjusted"""


class CannotSplitAcrossBond(AutodeException):
    """A molecule cannot be partitioned by deleting one bond"""


class CouldNotPlotSmoothProfile(AutodeException):
    """A smooth reaction profile cannot be plotted"""


class TemplateLoadingFailed(AutodeException):
    """A template file was not in the correct format"""


class OptimiserStepError(AutodeException):
    """Unable to calculate a valid Optimiser step"""


class CoordinateTransformFailed(AutodeException):
    """Internal coordinate to Cartesian transform failed"""
