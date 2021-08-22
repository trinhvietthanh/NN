import numpy as np
from NN import module

class Layer(module.Module):
    def __init__(self, trainable=True, name=None, dtype=None, **kwargs):
        allowed_kwargs = {
            'input_dim',
            'input_shape',
            'batch_input_shape',
            'batch_size',
            'weights',
            'activity_regularizer',
            'autocast',
            'implementation',
        }
        # self._init_set_name(name)
        self._trainable= trainable
    def add_weight(self, shape=None, dtype=None, initializer=None, trainable=None):

        if shape is None:
            shape=()
        if dtype is None:
            dtype = self.dtype
        dtype = dtypes.as_dtype(dtype)
        variable = self._add_variable_            
    
    # def build(self, input_shape):
    #     if 
    
    def  losses(self):
        collected_losses = []
        pass
    def add_loss(self, losses, **kwargs):
        kwargs.pop('inputs', None)
        if kwargs:
            raise TypeError('Unknown keyword arguments: %s' % (kwargs.keys(),))
        pass


class SimpleDense(Layer):
    def __init__(self, units=32) -> None:
        super(SimpleDense).__init__()
        self.units = units

    def build(self, input_shape):
        self.w = np.random.rand(input_shape[-1], self.units)
        self.b = np.random.rand(self.units,)

    def __call__(self, inputs):
        return inputs.dot(self.w) + self.b
   