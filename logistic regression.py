
########## input file "logistic_and_svm_data.txt" has changed with space.
import math
import numpy as np
from chart import Chart
from readfile import Readfile


FILE_NAME = "data/logistic_and_svm_data.txt"
SMALL_NUMBER = 0.000000000000001
##################################################
ITERATION_NUMBER = 15000
ALPHA = 0.0009
INPUT_NUMBER = 2
WEIGHT = [-4.0, 1.0, 1.0]
LAMBDA = 0.00001
#  W0 + W1*X1 + W2*X2
##################################################


def sigmoid(x):
    return 1.0/(1.0+np.exp(-x))


def hypothesis(weight, x, j):
    result = 0.0
    result += weight[0]
    for i in range(INPUT_NUMBER):
        result += (weight[i+1] * x[j][i])
    return sigmoid(result)


def summation(y, x, weight, hasX, m, k=-1):  # hasX for W1.
    loss = 0.0
    for i in range(m):
        hw = hypothesis(weight, x, i)
        error = (hw - y[i])
        if hasX:
            error = error * x[i][k]
        loss += error
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
    for i in range(m):
        cost1 = (-1) * y[i] * math.log(hypothesis(weight, x, i))
        if hypothesis(weight, x, i) != 1:  # for math error domain -> log(0) is not defined.
            cost2 = (1-y[i]) * math.log(1-hypothesis(weight, x, i))
        else:
            cost2 = (1-y[i]) * math.log(SMALL_NUMBER)
        cost += (cost1 - cost2)
    constant = float(1) / float(m)
    cost *= constant
    print "Cost function is:", cost


def logistic_regression(x, y, m, weight):
    for i in range(ITERATION_NUMBER):
        weight = gradient_descent(x, y, weight, m)
        cost_function(x, y, weight, m)
    return weight


def seperate_data(x, y, output, m):
    isSickX = []  # 1
    isSickY = []  # 1
    notSickX = []
    notSickY = []
    for i in range(m):
        if output[i] == 0:
            notSickX.append(x[i])
            notSickY.append(y[i])
        else:
            isSickX.append(x[i])
            isSickY.append(y[i])
    return isSickX, isSickY, notSickX, notSickY


if __name__ == '__main__':
    readFile = Readfile(FILE_NAME)

    x = readFile.readDataCol(1)
    y = readFile.readDataCol(2)
    output = readFile.readDataCol(3)
    isSickX, isSickY, notSickX, notSickY = seperate_data(x, y, output, len(output))
    input = readFile.readDataLines()

    weight = logistic_regression(input, output, len(input), WEIGHT)

    chart = Chart()
    chart.draw_classification_line(weight, notSickX, notSickY, isSickX, isSickY, input)
