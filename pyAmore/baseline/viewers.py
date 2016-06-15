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
                  ' Neuron: {label}\n'
                  '-----------------------------------\n'
                  ' Output: {output}\n'
                  '-----------------------------------\n'
                  ' Neuron predict strategy: {predict_strategy}'
                  '\n-----------------------------------\n'
                  ' Neuron fit strategy: {fit_strategy}'
                  '\n-----------------------------------'
                  '\n Connections:'
                  '\n-----------------------------------\n'
                  ).format(label=neuron.label,
                           output=round(neuron.output, 8),
                           predict_strategy=neuron.predict_strategy.__class__,
                           fit_strategy=neuron.fit_strategy.__class__,
                           )
        for connection in neuron.connections:
            result += ' Neuron: {label}, weight: {weight}\n'.format(label=connection.neuron.label,
                                                                    weight=connection.weight)
        result += '-----------------------------------\n'
        print(result)
        return result

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
        for neuron in neural_network.layers[0]:
            result += MlpNeuralViewer.show_neuron(neuron)

        result += ('\n----------------------------------------------\n'
                   '     HIDDEN LAYERS:\n'
                   '----------------------------------------------\n'
                   )
        for layer in neural_network.layers[1:-1]:
            result += '##############################################'
            for neuron in layer:
                result += MlpNeuralViewer.show_neuron(neuron)

        result += ('\n----------------------------------------------\n'
                   '     OUTPUT LAYER:\n'
                   '----------------------------------------------\n'
                   )
        for neuron in neural_network.layers[-1]:
            result += MlpNeuralViewer.show_neuron(neuron)
        print(result)
        return result
