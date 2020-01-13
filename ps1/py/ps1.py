# %% data for problem #1a
# clear
#
# a = 0
# b = 10
# m = 1000
# n = 1
#
# generate mXn matrix of continuous uniform
# random values between a and b
#
# x = unifrnd(a,b,m,n);
# e = normrnd(0,1,size(x));
# y = 2*x
# plot(x,y,'.');

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1)

def part_a():
    """
    MATLAB CODE:
    x = unifrnd(0,10,1000,1);
    y = 2*x + normrnd(0,1,size(x));
    plot(x,y,'.');

    line-by-line translation with same output, modified to fit the assignment
    """

    x = np.random.uniform(low=-10, high=10, size=1000)
    y = 2*x + np.random.normal(0, 1, x.size)
    plt.plot(x,y,".")

    plt.show()

def main():
    part_a()

if __name__ == '__main__':
    main()
