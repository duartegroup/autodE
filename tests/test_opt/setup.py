from autode.wrappers.methods import Method as BaseMethod
from autode.wrappers.keywords.keywords import KeywordsSet


class Method(BaseMethod):
    def __init__(self):
        super().__init__(
            name="test_method", keywords_set=KeywordsSet(), doi_list=[]
        )

    def __repr__(self):
        return f"{self.__class__.__name__}"

    def implements(
        self, calculation_type: "autode.calculations.types.CalculationType"
    ) -> bool:
        return True

    @property
    def uses_external_io(self) -> bool:
        return False
