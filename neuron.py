import container  # , connection

"""
#import  neuralFactory  , activation_function, predict_behavior
"""


class Neuron:
    def __init__(self):
        self.id = None
        self.induced_local_field = 0.0
        self.output = 0.0
        self.target = 0.0
        self.connections = container.Container()


class SimpleNeuron(Neuron):
    def __init__(self):
        Neuron.__init__(self)

    def add_connection(self, connection):
        self.connections.append(connection)

    def show(self):
        print("\n\n-----------------------------------")
        print("\n Id: {id}".format(id=self.id))
        print("\n-----------------------------------")
        print("\n output: {output}".format(output=self.output))
        print("\n-----------------------------------")
        #   self.predictBehavior.show()
        print("\n target: {target}".format(target=self.target))
        print("\n-----------------------------------")
        repr(self.connections)
        print("\n-----------------------------------")
        #   print("\n Neuron Train Behavior: %s", getNeuronTrainBehaviorName().c_str())
        #   print("\n-----------------------------------")


"""
double
SimpleNeuron::useActivationFunctionf0()
{
  return d_activationFunction->f0();
}


double
SimpleNeuron::useActivationFunctionf1()
{
  return d_activationFunction->f1();
}




"""

"""
bool
SimpleNeuron::validate()
{

BEGIN_RCPP
  if (getId() == NA_INTEGER ) throw std::range_error("[C++ SimpleNeuron::validate]: Error, Id is NA.");
    d_nCons->validate();
  return (TRUE);
END_RCPP
}






double
SimpleNeuron::costFunctionf0(double output, double target)
{
  return d_neuralNetwork->costFunctionf0( output, target );
}


double
SimpleNeuron::costFunctionf1(double output, double target)
{
  return d_neuralNetwork->costFunctionf1( output, target );
}



void
SimpleNeuron::addToBias(double value)
{
  d_predictBehavior->addToBias(value);
}


void
SimpleNeuron::addToDelta(double value)
{
  d_neuronTrainBehavior->addToDelta(value);

}

void
SimpleNeuron::setLearningRate(double learningRate)
{
  d_neuronTrainBehavior->setLearningRate(learningRate);
}
"""
