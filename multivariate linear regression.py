from chart import Chart
from readfile import Readfile

FILE_NAME = "data/housing.data.txt"
DATA_NUMBER = 506
##############################################
INPUT_NUMBER = 13
ITERATION_NUMBER = 6000  ##even more.
ALPHA = 0.0032
WEIGHT = [3.0, 1.0, 1.0, 2.0, 2.0, 1.0, 2.0, 1.0, 4.0, 3.0, 1.0, 3.0, 2.0, 1.0]
LAMBDA = 0.00001
# Y = W0 + W1*X1 + ... + W13*X13
##############################################


def hypothesis(weight, x, j):
    result = 0.0
    result += weight[0]
    for i in range(INPUT_NUMBER):
        result += (weight[i+1] * x[j][i])
    return result


def summation(y, x, weight, hasX, m, k=-1):  # hasX for W1.
    loss = 0.0
    for i in range(DATA_NUMBER):
        hw = hypothesis(weight, x, i)
        error = (hw - y[i])
        if hasX:
            error = error * x[i][k]
        loss += error
    const = float(1) / float(m)
    loss *= const
    return loss


def l2_norm(weights):
    result = []
    for i in weights:
        result.append(i - (LAMBDA * i))
    return result


def gradient_descent(x, y, weight, m):
    new_weight = []
    const = float(ALPHA)/float(m)
    new_weight.append(weight[0] - (const * summation(y, x, weight, False, m)))
    for i in range(INPUT_NUMBER):
        new_weight.append(weight[i+1] - (const * summation(y, x, weight, True, m, i)))
    new_weight = l2_norm(new_weight)  # l2-norm
    return new_weight


def cost_function(x, y, weight, m):  # cost function
    cost = 0.0
    for i in range(DATA_NUMBER):
        cost += (y[i] - hypothesis(weight, x, i))**2
    constant = float(1) / float(2*m)
    cost *= constant
    print "Cost function is:", cost


def mean_error(x, y, weight, m):  # mean error
    error = 0.0
    for i in range(DATA_NUMBER):
        error += (y[i] - hypothesis(weight, x, i))
    constant = float(1) / float(2 * m)
    error *= constant
    print "Mean error is:", error


def multivariante_linear_regression(x, y, weight):
    m = len(x)
    for i in range(ITERATION_NUMBER):
        weight = gradient_descent(x, y, weight, m)
        cost_function(x, y, weight, m)
        mean_error(x, y, weight, m)
    return weight


if __name__ == '__main__':
    readFile = Readfile(FILE_NAME)
    input = []
    input = readFile.readDataLines()
    price = readFile.readDataCol(14)

    weights = multivariante_linear_regression(input, price, WEIGHT)
    predict = []
    x_label = []
    for i in range(len(input)):
        predict.append(hypothesis(weights, input, i))
        x_label.append(i)

    chart = Chart()
    chart.draw_one_set_data(x_label, price, "number", "Sale", False, predict)
