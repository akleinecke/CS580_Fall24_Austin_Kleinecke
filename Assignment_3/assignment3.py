#!/usr/bin/env python3
import sys
import pandas as pand
import numpy as nump
import matplotlib.pyplot as plot


def load_data(path):
    # Load data
    data = pand.read_csv(path, header=None)
    x = data[0] # Column 1
    y = data[1] # Column 2

    return (x, y)


def calculate_slope(x, y):
    # Calculate means
    x_mean = nump.mean(x)
    y_mean = nump.mean(y)

    # Calculate covariance and variance
    cov_xy = nump.sum((x - x_mean) * (y - y_mean)) / len(x)
    var_x = nump.sum((x - x_mean) ** 2) / len(x)

    # Compute the slope and intercept
    m = cov_xy / var_x
    b = y_mean - m * x_mean

    return (m, b)



def plot_regression(x, y, m, b):
    y_prediction = m * x + b

    # Plot data and line
    plot.scatter(x, y, label='Data Points')
    plot.plot(x, y_prediction, color='red', label='Regression')
    plot.title('Linear Regression')
    plot.legend()
    plot.show()
    

def main():
    # Check for an additional command-line argument.
    # If found, it should ba another file we can open
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    else:
        file_path = 'linear_regression_data.csv' # Default

    xy = load_data(file_path)
    mb = calculate_slope(xy[0], xy[1])
    plot_regression(xy[0], xy[1], mb[0], mb[1])


if __name__ == "__main__":
    main()
