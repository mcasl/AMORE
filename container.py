class Container():
    """
    Interface for generic collection of items, ie. Conexions, Neurons, Layers, ...
    Currently though, it is nothing but a list wrapper.
    """
    def __init__(self):
        self.data = []

    def __repr__(self):
        return str(self.data)

    def __len__(self):
        return len(self.data)

    def __reversed__(self):
        return reversed(self.data)

    def __getattr__(self, attr):
        return getattr(self.data, attr)  # Delegate all other attrs
