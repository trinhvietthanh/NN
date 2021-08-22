import numpy as np
from NN.base_layer import Layer

class Model(Layer):
    def __init__(self, *args, **kwargs):
        super(Model, self).__init__(**kwargs)
        self.inputs = None
        self.outputs = None
        self.input_names = None
        self.output_names = None
        # self.compiled_loss = None
        # self.compiled_metrics = None
    def __call__(self, inputs, training=None, mask=None):
        pass
    
    def fit():
        pass
    def compile():
        pass

    def result():
        pass