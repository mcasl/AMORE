from common cimport RealNumber
cimport numpy as np
from .materials cimport MlpNetwork

cpdef MlpNetwork mlp_network(list layers_size,
                             str hidden_layers_activation_function_name,
                             str output_layer_activation_function_name)

cpdef MlpNetwork fit_adaptive_gradient_descent(MlpNetwork mlp_neural_network,
                                               np.ndarray input_data,
                                               np.ndarray target_data,
                                               RealNumber learning_rate,
                                               int step_length,
                                               int number_of_steps)
