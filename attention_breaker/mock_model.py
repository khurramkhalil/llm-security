import numpy as np

class MockModel:
    def __init__(self):
        self.layers = [{'name': 'layer1', 'weights': np.random.randint(0, 256, size=100)},
                       {'name': 'layer2', 'weights': np.random.randint(0, 256, size=100)}]

    def set_weights(self, layer_name, weights):
        for layer in self.layers:
            if layer['name'] == layer_name:
                layer['weights'] = weights

    def get_weights(self, layer_name):
        for layer in self.layers:
            if layer['name'] == layer_name:
                return layer['weights']

    def evaluate_loss(self):
        # Simple mock loss calculation
        return np.random.random()