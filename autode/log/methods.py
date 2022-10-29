class ComputationalMethods:
    """Generic computational methods section built by autodE"""

    def __init__(self):
        """
        ComputationalMethods as a list of sentences and digital object
        identifiers (DOIs) for all the methods used
        """
        self._list = []

    def __add__(self, other):
        """Add another sentence to the methods if it's not already present"""
        assert type(other) is str

        if other not in self._list:
            self._list.append(other)

        return None

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        """String of the computational methods used in this initialisation
        of autodE prepended with the current version"""
        import autode

        autode_str = (
            f"All calculations were performed in autodE "
            f"v. {autode.__version__} (10.1002/anie.202011941). "
        )

        return autode_str + " ".join(self._list)

    def add(self, other):
        """Add a string to the methods"""
        return self.__add__(other)

    def clear(self):
        """Clear the current string"""
        self._list = []
        return None


methods = ComputationalMethods()
