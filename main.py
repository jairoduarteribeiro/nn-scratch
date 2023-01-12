import numpy as np
import nnfs
import matplotlib.pyplot as plt
from nnfs.datasets import spiral_data

nnfs.init()


def main():
    x, y = spiral_data(samples=100, classes=3)
    plt.scatter(x[:, 0], x[:, 1], c=y, cmap='brg')
    plt.show()


if __name__ == '__main__':
    main()
