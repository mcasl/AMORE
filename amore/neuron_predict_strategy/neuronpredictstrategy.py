from abc import ABCMeta, abstractmethod


class NeuronPredictStrategy(object, metaclass=ABCMeta):
    @abstractmethod
    def __init__(self, neuron):
        self.neuron = neuron

    @abstractmethod
    def __call__(self, *args, **kwargs):
        raise NotImplementedError("You shouldn't be calling NeuronPredictStrategy.__call__")
