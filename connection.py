import neuron


class Connection:
    def __init__(self, target_neuron, weight_value=0.0):
        self.weight = weight_value
        self.neuron = target_neuron

    def id(self):
        return neuron.id

    def add_to_weight(self, value):
        self.weight += value;

    def add_to_delta(self, value):
        self.neuron.addToDelta(value)

    def input_value(self):
        return self.neuron.output

    def show(self):
        print('\nFrom:\t {id} \t Weight= \t {weight}'.format(id=self.id, weight=self.weight))

        def validate(self):
            pass

            # if (! R_FINITE(getWeight()) ) throw std::range_error("weight is not finite.");

            #          if (Id() == NA_INTEGER)
            #            throw std::range_error("fromId is not finite.");
            #          return (true);
