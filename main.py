import numpy as np
import nnfs
from nnfs.datasets import spiral_data
from layer_dense import LayerDense
from activation import ActivationReLU
from loss import ActivationSoftmaxLossCategoricalCrossEntropy

nnfs.init()


def main():
    x, y = spiral_data(samples=100, classes=3)
    dense1 = LayerDense(2, 3)
    activation1 = ActivationReLU()
    dense2 = LayerDense(3, 3)
    loss_activation = ActivationSoftmaxLossCategoricalCrossEntropy()
    dense1.forward(x)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    loss = loss_activation.forward(dense2.output, y)
    print(loss_activation.output[:5])
    print('loss:', loss)
    predictions = np.argmax(loss_activation.output, axis=1)
    if len(y.shape) == 2:
        y = np.argmax(y, axis=1)
    accuracy = np.mean(predictions == y)
    print('acc:', accuracy)
    loss_activation.backward(loss_activation.output, y)
    dense2.backward(loss_activation.d_inputs)
    activation1.backward(dense2.d_inputs)
    dense1.backward(activation1.d_inputs)
    print(dense1.d_weights)
    print(dense1.d_biases)
    print(dense2.d_weights)
    print(dense2.d_biases)


if __name__ == '__main__':
    main()
