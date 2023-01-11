import numpy as np


def main():
    inputs = [1, 2, 3, 2.5]
    weights = [0.2, 0.8, -0.5, 1]
    bias = 2
    outputs = np.dot(weights, inputs) + bias
    print(outputs)


if __name__ == '__main__':
    main()
