import nnfs
from nnfs.datasets import spiral_data
from layer_dense import LayerDense
from activation_relu import ActivationReLU

nnfs.init()


def main():
    x, y = spiral_data(samples=100, classes=3)
    dense1 = LayerDense(2, 3)
    activation1 = ActivationReLU()
    dense1.forward(x)
    activation1.forward(dense1.output)
    print(activation1.output[:5])


if __name__ == '__main__':
    main()
