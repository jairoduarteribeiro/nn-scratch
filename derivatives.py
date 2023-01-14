import numpy as np
import matplotlib.pyplot as plt


def f(x):
    return 2 * x


def main():
    x = np.array(range(5))
    y = f(x)
    plt.plot(x, y)
    plt.show()


if __name__ == '__main__':
    main()
