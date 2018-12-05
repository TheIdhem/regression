from numpy import array
import matplotlib.pyplot as plt
import numpy as np


class Chart:

    def draw_one_set_data(self, x, y, x_label, y_label, isLine, weight=None):  # for multivariate ln
        plt.scatter(x, y, color="green", marker="o", s=10)
        if weight != None and isLine:
            f = weight[1] * np.asarray(x) + weight[0]
            plt.plot(x, f, color="black")
        if weight != None and not(isLine):
            plt.scatter(x, weight, color="red", marker="o", s=10)  # our predict in multivariate.

        plt.xlabel(x_label)
        plt.ylabel(y_label)

        plt.show()


    def draw_classification_line(self, w, x1, y1, x2, y2, input):  ## for logistic regression
        plt.scatter(x1, y1, color="green", marker="o", s=10)  # not Sick
        plt.scatter(x2, y2, color="red", marker="o", s=10)  # Sick

        temp_input = np.array(input)
        plot_x = array([min(temp_input[:, 0]) - 2, temp_input[:, 1].max() + 2])
        f = (w[0] + w[1] * plot_x) * (-1/w[2])
        plt.plot(plot_x, f)

        plt.show()


    def draw_smv(self, x1, y1, x2, y2, w, b, maxX, minX):
        def f(x, w, b):
            return (-w[0] * x - b) / w[1]

        plt.scatter(x1, y1, color="green", marker="o", s=10)
        plt.scatter(x2, y2, color="red", marker="o", s=10)

        # # w.x + b = 0
        a0 = minX
        a1 = f(a0, w, b)
        b0 = maxX
        b1 = f(b0, w, b)
        plt.plot([a0, b0], [a1, b1], "k")

        plt.show()
