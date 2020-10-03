

class ComputationalMethods:
    """Generic computational methods section built by autodE"""

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
        autode_str = (f'All calculations were performed with autodE '
                      f'v. {autode.__version__}. ')

        return autode_str + " ".join(self._list)

    def clear(self):
        """Clear the current string"""
        self._list = []
        return None

    def string(self):
        return self.__str__()

    def __init__(self):
        """
        ComputationalMethods containing a list of digital object identifiers
        for all the methods used and a private list of strings for all the
        methods used
        """
        self._list = []


methods = ComputationalMethods()
