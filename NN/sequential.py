from NN.module import Module
from NN import base_layer
from NN import functional
import inspect

class Sequential(functional.Functional):
    """['Sequential' groups a linear stack of layers into model define]

    Args:
        functional ([type]): [description]
    """
    def __init__(self, layers=None, name=None):
        if layers:
            if not isinstance(layers, (list, tuple)):
                layers = [layers]
            for layer in layers:
                self.add(layer)
        self.layers = {}

    def add(self, layer):
        if not isinstance(layer, Module):
            raise TypeError('The added layer must be '
                      'an instance of class Layer. '
                      'Found: ' + str(layer))
        self.layers[layer] = inspect.getfullargspec(layer)

    def call(self, inputs, training=None):
        outputs = inputs # handle the corner case where self.layers is empty
        for layer in self.layers:
            # During each iteration, `inputs` are the inputs to `layer`, and `outputs`
            # are the outputs of `layer` applied to `inputs`. At the end of each
            # iteration `inputs` is set to `outputs` to prepare for the next layer.
            kwargs = {}
            
            kwargs['mask'] = None
            kwargs['training'] = training

            outputs = layer(inputs, **kwargs)
            inputs = outputs
        return outputs

        


