import numpy as np
import nnfs
from nnfs.datasets import spiral_data
from layer_dense import LayerDense
from activation import ActivationReLU
from loss import ActivationSoftmaxLossCategoricalCrossEntropy
from optimizer import OptimizerSGD

nnfs.init()


def main():
    x, y = spiral_data(samples=100, classes=3)
    dense1 = LayerDense(2, 64)
    activation1 = ActivationReLU()
    dense2 = LayerDense(64, 3)
    loss_activation = ActivationSoftmaxLossCategoricalCrossEntropy()
    optimizer = OptimizerSGD(learning_rate=0.85)
    for epoch in range(10001):
        dense1.forward(x)
        activation1.forward(dense1.output)
        dense2.forward(activation1.output)
        loss = loss_activation.forward(dense2.output, y)
        predictions = np.argmax(loss_activation.output, axis=1)
        if len(y.shape) == 2:
            y = np.argmax(y, axis=1)
        accuracy = np.mean(predictions == y)
        if not epoch % 100:
            print(f'epoch: {epoch}, acc: {accuracy:.3f}, loss: {loss:.3f}')
        loss_activation.backward(loss_activation.output, y)
        dense2.backward(loss_activation.d_inputs)
        activation1.backward(dense2.d_inputs)
        dense1.backward(activation1.d_inputs)
        optimizer.update_params(dense1)
        optimizer.update_params(dense2)


if __name__ == '__main__':
    main()
