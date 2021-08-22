import numpy as np
from NN.base_layer import Layer
from NN import activations

class Dense(Layer):
    def __init__(self, units, activation=None, use_bias=True, **kwargs):
        super(Dense, self).__init__(**kwargs)
        self.units = int(units)
        self.activation = activations.get(activation)
        self.units = int(units) if not isinstance(units, int) else units
        if self.units < 0:
            raise ValueError(f'Invalid value for `units`')
        self.activation = activations.get(activation)
        self.use_bias = use_bias
    
    def __call__(self, inputs, **kwargs):
        weight = np.random.rand(len(inputs), self.units)
        outputs = inputs.dot(weight)
        if self.use_bias:
            bias = np.random.rand(self.units,)
            outputs += bias     
        if self.activation is not None:
            result = []
            for i in outputs:
                result.append(self.activation(i))
            outputs = np.array(result)
        return outputs

