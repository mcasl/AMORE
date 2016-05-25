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


class MlpNeuralViewer(NeuralViewerConsole):
    @staticmethod
    def show_connection(connection):
        print('\nFrom:\t {label} \t Weight= \t {weight}'.format(label=connection.neuron.label,
                                                                weight=connection.weight))

    @staticmethod
    def show_neuron(neuron):
        result = ('\n\n'
                  '-----------------------------------\n'
                  ' Id label: {label}\n'
                  '-----------------------------------\n'
                  ' Output: {output}\n'
                  '-----------------------------------\n'
                  # TODO:    '{predict_behavior}'
                  ' Target: {target}\n'
                  '-----------------------------------\n').format(label=neuron.label,
                                                                  output=neuron.output,
                                                                  # TODO:  predict_behavior=repr(self.predictBehavior),
                                                                  target=neuron.target)
        result += repr(neuron.connections)
        #          '\n-----------------------------------\n'
        #                   'Neuron Train Behavior: {train_behavior}'.format(train_behavior=self.train_behavior),
        #         '\n-----------------------------------'
        print(result)

    @staticmethod
    def show_neural_network(neural_network):
        """ Pretty print
        """
        result = ('\n----------------------------------------------\n'
                  'Simple Neural Network\n'
                  '----------------------------------------------\n'
                  '     INPUT LAYER:\n'
                  '----------------------------------------------\n'
                  )
        result += repr(neural_network.layers[0])
        result += ('\n----------------------------------------------\n'
                   '     HIDDEN LAYERS:\n'
                   '----------------------------------------------\n'
                   )
        result += repr(neural_network.layers[1:-1])
        result += ('\n----------------------------------------------\n'
                   '     OUTPUT LAYER:\n'
                   '----------------------------------------------\n'
                   )
        result += repr(neural_network.layers[-1])
        print(result)
