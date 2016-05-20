from abc import ABCMeta, abstractmethod


class NeuralViewerConsole(object, metaclass=ABCMeta):
    @staticmethod
    @abstractmethod
    def show_connection(connection):
        pass

    @staticmethod
    @abstractmethod
    def show_neuron(neuron):
        pass

    @staticmethod
    @abstractmethod
    def show_neural_network(neural_network):
        pass
