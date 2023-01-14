import nnfs
from nnfs.datasets import spiral_data
from layer_dense import LayerDense
from activation import ActivationReLU, ActivationSoftmax
from loss import LossCategoricalCrossEntropy

nnfs.init()


def main():
    x, y = spiral_data(samples=100, classes=3)
    dense1 = LayerDense(2, 3)
    activation1 = ActivationReLU()
    dense2 = LayerDense(3, 3)
    activation2 = ActivationSoftmax()
    loss_function = LossCategoricalCrossEntropy()
    dense1.forward(x)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    activation2.forward(dense2.output)
    print(activation2.output[:5])
    loss = loss_function.calculate(activation2.output, y)
    print(f'loss: {loss}')


if __name__ == '__main__':
    main()
