class Container:
    """
    A generic collection of items, eg. Connections, Neurons, Layers, ...
    Currently though, it is nothing but a list wrapper.
    """

    def __init__(self, iterable=None):
        self.data = []
        if iterable is not None:
            self.extend(iterable)

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

    def __eq__(self, other):
        return self.data == other

    def __ne__(self, other):
        return self.data != other

    def __getattr__(self, attr):
        return getattr(self.data, attr)  # Delegate  those  attributes not overridden
