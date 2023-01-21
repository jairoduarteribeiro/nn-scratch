import numpy as np
from abc import ABC, abstractmethod


class Optimizer(ABC):
    def __init__(self, learning_rate=1.0, decay=0.0):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0

    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = \
                self.learning_rate * (1.0 / (1.0 + self.decay * self.iterations))

    @abstractmethod
    def update_params(self, layer):
        pass

    def post_update_params(self):
        self.iterations += 1


class OptimizerSGD(Optimizer):
    def __init__(self, learning_rate=1.0, decay=0.0, momentum=0.0):
        super().__init__(learning_rate, decay)
        self.momentum = momentum

    def update_params(self, layer):
        if self.momentum:
            if not hasattr(layer, 'weight_momentums'):
                layer.weight_momentums = np.zeros_like(layer.weights)
                layer.bias_momentums = np.zeros_like(layer.biases)
            weight_updates = \
                self.momentum * layer.weight_momentums - \
                self.current_learning_rate * layer.d_weights
            layer.weight_momentums = weight_updates
            bias_updates = \
                self.momentum * layer.bias_momentums - \
                self.current_learning_rate * layer.d_biases
            layer.bias_momentums = bias_updates
        else:
            weight_updates = -self.current_learning_rate * layer.d_weights
            bias_updates = -self.current_learning_rate * layer.d_biases
        layer.weights += weight_updates
        layer.biases += bias_updates


class OptimizerAdaGrad(Optimizer):
    def __init__(self, learning_rate=1.0, decay=0.0, epsilon=1e-7):
        super().__init__(learning_rate, decay)
        self.epsilon = epsilon

    def update_params(self, layer):
        if not hasattr(layer, 'weight_cache'):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)
        layer.weight_cache += layer.d_weights ** 2
        layer.bias_cache += layer.d_biases ** 2
        layer.weights += \
            -self.current_learning_rate * layer.d_weights / \
            (np.sqrt(layer.weight_cache) + self.epsilon)
        layer.biases += \
            -self.current_learning_rate * layer.d_biases / \
            (np.sqrt(layer.bias_cache) + self.epsilon)


class OptimizerRMSProp(OptimizerAdaGrad):
    def __init__(self, learning_rate=1.0, decay=0.0, epsilon=1e-7, rho=0.9):
        super().__init__(learning_rate, decay, epsilon)
        self.rho = rho

    def update_params(self, layer):
        if not hasattr(layer, 'weight_cache'):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)
        layer.weight_cache = \
            self.rho * layer.weight_cache + (1 - self.rho) * layer.d_weights ** 2
        layer.bias_cache = \
            self.rho * layer.bias_cache + (1 - self.rho) * layer.d_biases ** 2
        layer.weights += \
            -self.current_learning_rate * layer.d_weights / \
            (np.sqrt(layer.weight_cache) + self.epsilon)
        layer.biases += \
            -self.current_learning_rate * layer.d_biases / \
            (np.sqrt(layer.bias_cache) + self.epsilon)


class OptimizerAdam(OptimizerAdaGrad):
    def __init__(self, learning_rate=0.001, decay=0.0, epsilon=1e-7, beta_1=0.9, beta_2=0.999):
        super().__init__(learning_rate, decay, epsilon)
        self.beta_1 = beta_1
        self.beta_2 = beta_2

    def update_params(self, layer):
        if not hasattr(layer, 'weight_cache'):
            layer.weight_momentums = np.zeros_like(layer.weights)
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_momentums = np.zeros_like(layer.biases)
            layer.bias_cache = np.zeros_like(layer.biases)
        layer.weight_momentums = \
            self.beta_1 * layer.weight_momentums + (1 - self.beta_1) * layer.d_weights
        layer.bias_momentums = \
            self.beta_1 * layer.bias_momentums + (1 - self.beta_1) * layer.d_biases
        weight_momentums_corrected = \
            layer.weight_momentums / (1 - self.beta_1 ** (self.iterations + 1))
        bias_momentums_corrected = \
            layer.bias_momentums / (1 - self.beta_1 ** (self.iterations + 1))
        layer.weight_cache = \
            self.beta_2 * layer.weight_cache + (1 - self.beta_2) * layer.d_weights ** 2
        layer.bias_cache = \
            self.beta_2 * layer.bias_cache + (1 - self.beta_2) * layer.d_biases ** 2
        weight_cache_corrected = \
            layer.weight_cache / (1 - self.beta_2 ** (self.iterations + 1))
        bias_cache_corrected = \
            layer.bias_cache / (1 - self.beta_2 ** (self.iterations + 1))
        layer.weights += \
            -self.current_learning_rate * weight_momentums_corrected / \
            (np.sqrt(weight_cache_corrected) + self.epsilon)
        layer.biases += \
            -self.current_learning_rate * bias_momentums_corrected / \
            (np.sqrt(bias_cache_corrected) + self.epsilon)
