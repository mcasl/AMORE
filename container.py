class Container:
    """
    Interface for generic collection of items, ie. Connections, Neurons, Layers, ...
    Currently though, it is nothing but a list wrapper.
    """

    def __init__(self, value=None):
        if value is None:
            self.data = []
        else:
            self.data = value

    def __getitem__(self, item):
        result = self.data[item]
        if isinstance(result, type(self.data)):
            result = Container(result)
        return result

    def __repr__(self):
        return repr(self.data)

    def __len__(self):
        return len(self.data)

    def __reversed__(self):
        return reversed(self.data)

    def __getattr__(self, attr):
        return getattr(self.data, attr)  # Delegate all other attributes --- those not found in __dict__
