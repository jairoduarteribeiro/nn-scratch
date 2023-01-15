import numpy as np
import matplotlib.pyplot as plt


def f(x):
    return 2 * x**2


def tangent_line(x, m, b):
    return x * m + b


def main():
    x = np.arange(0, 5, 0.001)
    y = f(x)
    colors = ('k', 'g', 'r', 'b', 'c')
    plt.plot(x, y)
    for i in range(5):
        p2_delta = 0.0001
        x1 = i
        x2 = x1 + p2_delta
        y1 = f(x1)
        y2 = f(x2)
        print((x1, y1), (x2, y2))
        approximate_derivative = (y2 - y1) / (x2 - x1)
        b = y2 - (approximate_derivative * x2)
        to_plot = [x1 - 0.9, x1, x1 + 0.9]
        plt.scatter(x1, y1, c=colors[i])
        plt.plot([point for point in to_plot],
                 [tangent_line(point, approximate_derivative, b) for point in to_plot],
                 c=colors[i])
        print(f'Approximate derivative for f(x) where x = {x1} is {approximate_derivative}')
    plt.show()


if __name__ == '__main__':
    main()
