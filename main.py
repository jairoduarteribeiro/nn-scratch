import nnfs
from nnfs.datasets import spiral_data
from layer_dense import LayerDense

nnfs.init()


def main():
    x, y = spiral_data(samples=100, classes=3)
    dense1 = LayerDense(2, 3)
    dense1.forward(x)
    print(dense1.output[:5])


if __name__ == '__main__':
    main()
