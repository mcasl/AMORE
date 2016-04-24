from abc import ABCMeta, abstractmethod

class Container(metaclass=ABCmeta)
    """
    Interface for generic collection of items, ie. Conexions, Neurons, Layers, ...
    """
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def __iter__(self):
        pass

    @abstractmethod
    def createReverseIterator(self):
        pass

    @abstractmethod
    def append(self, item):
        pass

    @abstractmethod
    def at(self, element):
        pass

    @abstractmethod
    def show(self):
        pass

    @abstractmethod
    def validate(self):
        pass


class SimpleContainer(Container):

    def __init__(self):
        self.collection = []

    def __iter__(self):
        return Simple_container_iterator(self)

    def createReverseIterator(self):
        return Simple_container_reverse_iterator(self)

    def at(self, element):
        return self.collection[element]

    def append(self, item):
        self.collection.append(item)

    def show(self):
        for item in self:
            item.show()

    def validate(self):
        for item in self:
            item.validate()
        return true
