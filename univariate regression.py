from chart import Chart
from readfile import Readfile

FILE_NAME = "data/housing.data.txt"
##############################################
ITERATION_NUMBER = 4000
ALPHA = 0.8
W0 = 5.0
W1 = 4.0
# Y = W1*X + W0
##############################################


def hypothesis(weight, x):
    return weight[1] * x + weight[0]


def summation(y, x, weight, hasX, m):  # hasX for W1.
    loss = 0.0
    for i in range(len(x)):
        hw = hypothesis(weight, x[i])
        error = (hw - y[i])
        if hasX:
            error = error * x[i]
        loss += error
    const = float(1) / float(m)
    loss *= const
    return loss


def gradient_descent(x, y, weight, m):
    new_weight = []
    const = float(ALPHA)/float(m)
    new_weight.append(weight[0] - (const * summation(y, x, weight, False, m)))
    new_weight.append(weight[1] - (const * summation(y, x, weight, True, m)))  # Xi
    return new_weight


def cost_function(x, y, weight, m):  # cost function
    cost = 0.0
    for i in range(m):
        cost += (y[i] - hypothesis(weight, x[i]))**2
    constant = float(1) / float(2*m)
    cost *= constant
    print "Cost function is:", cost


def mean_error(x, y, weight, m):  # mean error
    error = 0.0
    for i in range(m):
        error += (y[i] - hypothesis(weight, x[i]))
    constant = float(1) / float(2 * m)
    error *= constant
    print "Mean error is:", error


def linear_regression(x, y):
    weight = [W0, W1]
    m = len(x)
    for i in range(ITERATION_NUMBER):
        weight = gradient_descent(x, y, weight, m)
        cost_function(x, y, weight, m)
        mean_error(x, y, weight, m)
    return weight


if __name__ == '__main__':
    readFile = Readfile(FILE_NAME)
    crime_n = readFile.readDataCol(1)
    tax = readFile.readDataCol(3)
    price = readFile.readDataCol(14)

    chart = Chart()

    chart.draw_one_set_data(crime_n, price, "Crime", "Sale", False)
    weights = linear_regression(crime_n, price)
    chart.draw_one_set_data(crime_n, price, "Crime", "Sale", True, weights)

    chart.draw_one_set_data(tax, price, "Tax", "Sale", False)
    weights = linear_regression(tax, price)
    chart.draw_one_set_data(tax, price, "Tax", "Sale", True, weights)
