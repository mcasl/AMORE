class Connection(object):
    """ A simple data structure for linking neurons
    """

    def __init__(self, neuron, weight=0.0):
        """
        Initializer.
        :param neuron: Input neuron
        :param weight: float value
        :return: An initialized Connection instance
        """
        self.weight = weight
        self.neuron = neuron
