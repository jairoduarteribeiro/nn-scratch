import numpy as np
from abc import ABC, abstractmethod
from activation import ActivationSoftmax


class Loss(ABC):
    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        return data_loss

    @abstractmethod
    def forward(self, y_pred, y_true):
        pass


class LossCategoricalCrossEntropy(Loss):
    def __init__(self):
        self.d_inputs = None

    def forward(self, y_pred, y_true):
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples), y_true]
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)
        else:
            raise RuntimeError('Invalid y_true.shape')
        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods

    def backward(self, d_values, y_true):
        samples = len(d_values)
        labels = len(d_values[0])
        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]
        self.d_inputs = -y_true / d_values
        self.d_inputs = self.d_inputs / samples


class ActivationSoftmaxLossCategoricalCrossEntropy:
    def __init__(self):
        self.output = None
        self.d_inputs = None
        self.activation = ActivationSoftmax()
        self.loss = LossCategoricalCrossEntropy()

    def forward(self, inputs, y_true):
        self.activation.forward(inputs)
        self.output = self.activation.output
        return self.loss.calculate(self.output, y_true)

    def backward(self, d_values, y_true):
        samples = len(d_values)
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)
        self.d_inputs = d_values.copy()
        self.d_inputs[range(samples), y_true] -= 1
        self.d_inputs = self.d_inputs / samples
