from NN import model

class Functional(model.Model):
    def __init__(self, inputs, outputs, name=None, trainable=True):
        
        self._input_layers = []
        self._output_layers = []
        super(Functional, self).__init__(name=name, trainable=trainable)
        self._init_stack_network(inputs, outputs)
    
    def _init_stack_network(self, inputs, outputs):
        pass
