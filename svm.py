import numpy as np

import cvxopt
from scipy import linalg

from chart import Chart
from readfile import Readfile

FILE_NAME = "data/logistic_and_svm_data.txt"
######################################################
C = 0.1
INPUT_COLUMN = 2
######################################################


def linear_kernel(x1, x2):
    return np.dot(x1, x2)


def kernel_function(x):
    m = len(x)
    k = np.zeros((m, m))
    for i in range(m):
        for j in range(m):
            k[i, j] = linear_kernel(x[i], x[j])
    return k


def quadratic_programming(y, k, c):
    m = len(y)
    p = cvxopt.matrix(np.outer(y, y) * k)
    q = cvxopt.matrix(np.ones(m) * -1)
    a = cvxopt.matrix(y, (1, m))  # 90 ta 1
    b = cvxopt.matrix(0.0)

    # if c is None:
        # g = cvxopt.matrix(np.diag(np.ones(m) * -1))
        # h = cvxopt.matrix(np.zeros(m))
    # else:
    tmp1 = np.diag(np.ones(m) * -1)
    tmp2 = np.identity(m)
    g = cvxopt.matrix(np.vstack((tmp1, tmp2)))
    tmp1 = np.zeros(m)
    tmp2 = np.ones(m) * c
    h = cvxopt.matrix(np.hstack((tmp1, tmp2)))

    solution = cvxopt.solvers.qp(p, q, g, h, a, b)
    return np.ravel(solution['x'])


def supported_vector_data(alpha, x, y):
    sv = alpha > 1e-5  # check for supported vector points.
    ind = np.arange(len(alpha))[sv]
    sv_alpha = alpha[sv]
    sv_x = x[sv]
    sv_y = y[sv]
    return sv_alpha, sv_x, sv_y, ind, sv


def cal_coefficient(sv_alpha, sv_y, sv_x, k, ind, sv):
    # bias
    b = 0.0
    for n in range(len(sv_alpha)):
        b += sv_y[n]
        b -= np.sum(sv_alpha * sv_y * k[ind[n], sv])
    b /= len(sv_alpha)

    # Weight
    w = np.zeros(INPUT_COLUMN)
    for i in range(len(sv_alpha)):
        w += sv_alpha[i] * sv_y[i] * sv_x[i]

    return w, b


def svm(x, y, c):
    k = kernel_function(x)
    alpha = quadratic_programming(y, k, c)
    sv_alpha, sv_x, sv_y, ind, sv = supported_vector_data(alpha, x, y)
    weigth, b = cal_coefficient(sv_alpha, sv_y, sv_x, k, ind, sv)
    return weigth, b


def seperate_test_data(x, y, length):
    train_x = []
    train_y = []
    test_x = []
    test_y = []
    test_n = length/10
    for i in range(test_n):
        test_x.append(x[0])
        test_y.append(y[0])
        del x[0]
        del y[0]
    train_x = x
    train_y = y
    return train_x, train_y, test_x, test_y


def predict(test_x, test_y, w, b):
    predict_y = np.dot(test_x, w) + b
    return np.sign(predict_y)


def seperate_data(x, y, output, m):
    isSickX = []  # 1
    isSickY = []  # 1
    notSickX = []
    notSickY = []
    for i in range(m):
        if output[i] == -1:
            notSickX.append(x[i])
            notSickY.append(y[i])
        else:
            isSickX.append(x[i])
            isSickY.append(y[i])
    return isSickX, isSickY, notSickX, notSickY


def fix_data(temp_y):
    y = []
    for i in temp_y:
        if i == 0:
            y.append(float(-1))
        else:
            y.append(float(1))
    return y


def fix_x_data(x):
    result = []
    for i in x:
        temp = []
        for j in i:
            temp.append(j/100)
        result.append(temp)
    return result


if __name__ == '__main__':
    readFile = Readfile(FILE_NAME)

    x = readFile.readDataLines()
    temp_y = readFile.readDataCol(3)
    y = fix_data(temp_y)

    train_x, train_y, test_x, test_y = seperate_test_data(x[:], y[:], len(x))

    temp_train_x = np.array(train_x)
    temp_train_y = np.array(train_y)

    weight, bias = svm(temp_train_x, temp_train_y, C)
    predict_y = predict(test_x, test_y, weight, bias)

    counter = 0
    for i in range(len(test_y)):
        if predict_y[i] == test_y[i]:
            counter += 1
    print counter, "number of predictions are correct."

    x_points = readFile.readDataCol(1)
    y_points = readFile.readDataCol(2)

    maxX = max(x_points)
    minX = min(x_points)

    isSickX, isSickY, notSickX, notSickY = seperate_data(x_points, y_points, y, len(y))
    chart = Chart()
    chart.draw_smv(isSickX, isSickY, notSickX, notSickY, weight, bias, maxX, minX)
