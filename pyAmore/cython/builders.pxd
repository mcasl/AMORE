from materials cimport *

cdef class NetworkBuilder:
    cdef public:
        str label = ''

    cpdef Network create_neural_network(self, *args)

cdef class MlpNetworkBuilder(NetworkBuilder):
    cpdef MlpNetwork create_neural_network(self,
                                           neural_factory,  # TODO: MaterialsFactory neural_factory
                                           list layers_size,
                                           str hidden_layers_activation_function_name,
                                           str output_layer_activation_function_name)

    cpdef set_neurons_fit_strategy(self, neural_factory, MlpNetwork neural_network)

    cpdef set_neurons_predict_strategy(self, neural_factory, MlpNetwork neural_network)

    cpdef set_neurons_learning_rate(self, MlpNetwork neural_network, RealNumber learning_rate)

    cpdef create_primitive_layers(self, neural_factory, MlpNetwork neural_network, list layers_size)

    cpdef connect_network_layers(self, neural_factory, MlpNetwork neural_network)

    cpdef create_neuron_fit_and_predict_sequence(self, MlpNetwork neural_network)

    cpdef initialize_network(self, MlpNetwork neural_network)
