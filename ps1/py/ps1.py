#!/usr/bin/env python2

import numpy as np
import math
import matplotlib.pyplot as plt


np.random.seed(1)

def sigmoid(x):
    """
    args: x - some number
    return: some value between 0 and 1 based on sigmoid function
    """
    sigmoid = 1/(1+np.exp(-x))

    return sigmoid

def sigmoid_derivative(x):
    """
    args: x - some number
    return: derivative of sigmoid given x
    """
    sig_prime = x*(1-x)

    return sig_prime

def sign(x):
    """
    returns sign of x
    """
    return np.sign(x)

def sign_derivative(x):
    """
    returns "derivative" of sign
    """
    return x

def normalize(arr):
    """
    returns normalized array between -1 and 1
    """

    arr = arr / np.abs(arr).max(axis=0)

    return arr

def function(x):
    """
    mu = 0
    sigma = 1
    """
    y = 2*x + np.random.normal(0, 1, x.size) # [1000x1]

    return y

def part_1a():
    """
    MATLAB CODE:
    x = unifrnd(0,10,1000,1);
    y = 2*x + normrnd(0,1,size(x));
    plot(x,y,'.');

    line-by-line translation with same output, modified to fit the assignment

    returns x, y
    """

    x = np.random.uniform(low=-10, high=10, size=1000) # [1000x1]
    y = function(x) # [1000x1]

    fig1 = plt.figure()
    plt.plot(x, y, ".",label="sample points")
    plt.plot(x, 2*x, "r", label="original function")
    plt.title("Part A: Dataset")
    plt.xlabel("Random Uniform Distribution: -10 < n < 10")
    # plt.ylabel("2*x + e: mu = 0, sigma = 1,q size(x)")
    plt.legend()

    return x, y # [1000x1], [1000x1]


def part_1b(x):
    """
    Place their centers at -12 to 12 at every .5 along the x axis.
    Set the standard deviation of each RBF to 1.

    returns radial basis function h(x)
    """
    sigma = 1 # standard deviation
    x_u = np.linspace(-12, 12, 48) # centers at -12 to 12 @ every .5 [48x1]

    h = np.zeros((len(x),len(x_u))) # [1000x48]
    for j in range(len(x_u)): # 1-48
        for i in range(len(x)): # 1-1000
            h[i,j] = math.exp((-abs(x[i]-x_u[j])**2)/sigma**2)

    fig2 = plt.figure()
    plt.plot(x_u, h.T)#,":")
    plt.title("Part B: Radial Basis Functions")
    plt.xlabel("n = 48, -12 < n < 12")
    plt.ylabel("axis=0, mu = 0, sigma = 1, size(x)")
    plt.ylim(bottom = 0)
    plt.ylim(top = 1)
    # plt.legend()

    return h # [1000x48]


def part_1c(y, h):
    """
    calculated weight vector w using linear regression

    returns weight vector
    """
    # w = np.random.random((48)) # starting weights [48x1]
    w = np.dot(np.linalg.inv(np.dot(h.T,h)),np.dot(h.T,y))

    return w


def part_1d(x, y, h, w):
    """
    plots predicted f(x) given x, h(x), and w
    """
    f_x = np.random.random((1000))

    for i in range(len(f_x)):
        f_x[i] = sum(h[i]*w)

    fig3 = plt.figure()
    plt.plot(x, y, ".", label="true y values; n=1,000")
    plt.plot(x, f_x, ".", label="predicted y values; n=1,000")
    plt.plot(x, 2*x, "r", label="original function")
    plt.title("Part D: RBF Predictions")
    plt.legend()


def part_1e(x, y):
    """
    adds pertubation, simulated as 50 points at x=6
    creates new vectors of length 50, at x=6 and y=2x
    appends to existing x, y vectors

    returns x, y
    returns six_x, six_y for future graphing
    """
    six_x = np.arange(50, dtype=int) # creates a vector of length 50
    six_x = np.full_like(six_x, 6)
    six_y = 2*six_x + 10 + np.random.normal(0, 1, six_x.size) # [1050x1]

    x = np.append(x, six_x)
    y = np.append(y, six_y)

    fig4 = plt.figure()
    plt.plot(x, y, ".", label="sample points")
    plt.plot(six_x, six_y, 'g.', label="added points")
    plt.plot(x, 2*x, "r", label="original function")
    plt.title("Part E: Pertubation")
    plt.legend()

    return x, y, six_x, six_y


def part_1f(x, y, six_x, six_y):
    """
    plots predicted f(x) given x, h(x), and w

    returns new f(x)
    """
    sigma = 1 # standard deviation
    x_u = np.linspace(-12, 12, 48) # centers at -12 to 12 @ every .5 [48x1]

    h = np.zeros((len(x),len(x_u))) # [1000x48]
    for j in range(len(x_u)): # 1-48
        for i in range(len(x)): # 1-1000
            h[i,j] = math.exp((-abs(x[i]-x_u[j])**2)/sigma**2)

    w = np.dot(np.linalg.inv(np.dot(h.T,h)),np.dot(h.T,y))

    f_x = np.random.random((1050))

    for i in range(len(f_x)):
        f_x[i] = sum(h[i]*w)

    fig5 = plt.figure()
    plt.plot(x, y, ".", label="sample points; n=1,050")
    plt.plot(x, f_x, ".", label="predicted y values; n=1,050")
    plt.plot(six_x, six_y, '.', label="added points")
    plt.plot(x, 2*x, "r", label="original function")
    plt.title("Part F: RBF Predictions w/ Pertubation")
    plt.legend()

    return f_x


def part_1g(x, f_x):
    """
    finds mean error of all 1,050 samples
    normalizes values as percentile of normal distribution (mu = mean error, sigma = 1)
    """
    sigma = 1
    y = 2*x
    error = (y-f_x)**2
    mean_error = sum(error)/len(f_x)
    mean_error_arr = np.arange(len(x), dtype=int)
    mean_error_arr = np.full_like(mean_error_arr, mean_error)

    percent_error = (error-mean_error)/100

    fig6 = plt.figure()
    plt.plot(percent_error, label="percent error")
    plt.plot(mean_error_arr, 'r.', label="mean error = 0.044")
    plt.title("Part G: Error Analysis")
    plt.xlabel("samples; n=1,050")
    plt.ylabel("error normalized")
    plt.ylim(-0.1,1.0)
    plt.legend()

def part_2a():

    """
    **********************************************
    create training dataset
    **********************************************

    matlab dataset:

    dat1 = [normrnd(6,2,100,1) normrnd(2,1,100,1)];   % create the input data
    dat2 = [normrnd(2,3,100,1) normrnd(8,1,100,1)];
    dat = [dat1; dat2];

    y = [ones(100,1); -1*ones(100,1)];   % these are the labels for the classes
    """

    tempx = np.random.normal(6,2,size=100)
    tempy = np.random.normal(2,1,size=100)
    dat1 = np.array((tempx,tempy)).T

    tempx = np.random.normal(2,3,size=100)
    tempy = np.random.normal(8,1,size=100)
    dat2 = np.array((tempx,tempy)).T

    dat = np.append(dat1, dat2, axis=0) # [200x2]

    y = np.expand_dims(np.append(np.ones((100,1)),-1*np.ones((100,1))),axis=1) # [200x1]

    """
    shuffle dataset
    """
    temp = np.append(dat,y,axis=1)
    np.random.shuffle(temp)
    dat = temp[:,:2]
    tempy = temp[:,-1]
    y = np.expand_dims(tempy,axis=1)

    """
    **********************************************
    create validation dataset
    **********************************************
    """
    # v_tempx = np.random.normal(6,2,size=10)
    # v_tempy = np.random.normal(2,1,size=10)
    # v_dat1 = np.array((v_tempx,v_tempy)).T
    #
    # v_tempx = np.random.normal(2,3,size=10)
    # v_tempy = np.random.normal(8,1,size=10)
    # v_dat2 = np.array((v_tempx,v_tempy)).T
    #
    # v_dat = np.append(v_dat1, v_dat2, axis=0) # [200x2]
    #
    # y = np.expand_dims(np.append(np.ones((10,1)),-1*np.ones((10,1))),axis=1) # [200x1]
    #
    # """
    # shuffle dataset
    # """
    # v_temp = np.append(v_dat,v_y,axis=1)
    # np.random.shuffle(v_temp)
    # v_dat = v_temp[:,:2]
    # v_tempy = v_temp[:,-1]
    # v_y = np.expand_dims(v_tempy,axis=1)


    """
    **********************************************
    set network parameters
    **********************************************
    """
    epochs = 3000
    w = np.random.random((3,1)) # starting weight for each column (synapse)
    bias = np.ones((len(dat),1)) # bias
    error_arr = np.zeros((epochs,1))


    """
    **********************************************
    perceptron single layer network
    **********************************************
    """
    print("weights before: ")
    print w
    print
    # bias = 1 # bias
    bias = np.ones((len(dat),1)) # bias

    for i in range(epochs):
        """
        neuron
        """
        input = dat
        input = np.append(input,bias,axis=1)
        xw = np.dot(input,w) # [4x3]*[3*1]=[4x1]

        y_hat = sign(xw)

        error = y - y_hat
        error_arr[i] = sum(error)

        adjustments = error * sign_derivative(y_hat)

        w = w + np.dot(input.T,adjustments)
        # w = normalize(w)

    print "Weights after training: "
    print w
    print

    # print "percent error:"
    # percent_error=(sum(y-np.round(y_hat,0))/epochs)*100
    # print percent_error[0]
    # print

    y_hat = np.dot(input,w)

    """
    **********************************************
    plotting
    **********************************************
    """
    fiG0 = plt.figure()
    plt.plot(dat1.T[0], dat1.T[1], "ro", label="dat1")
    plt.plot(dat2.T[0], dat2.T[1], "bo", label="dat2")

    plt.title("Part 2a: 2 Classes, Initial Dataset")
    plt.xlim(-5,10)
    plt.ylim(-4,14)
    plt.legend()

    fig1 = plt.figure()
    plt.plot(dat1.T[0], dat1.T[1], "ro", label="dat1")
    plt.plot(dat2.T[0], dat2.T[1], "bo", label="dat2")
    # plt.plot(np.cross(w.T,input))
    # plt.plot((sum(y-np.round(y_hat,0))/epochs)*100,"wo",label="percent error: " + str(percent_error))# + str((sum(y-np.round(y_hat,0))/200)*100))
    for i in range(len(y_hat)):
    	if (y_hat[i]<0):
    		plt.plot(dat[i][0], dat[i][1],"b.")
    	else:
    		plt.plot(dat[i][0], dat[i][1],"r.")
    plt.title("Part 2a: Classification using sign(wx)")
    plt.xlim(-5,10)
    plt.ylim(-4,14)
    plt.legend()

    # fig2 = plt.figure()
    # plt.plot(abs(error_arr), label="Error")
    # plt.title("Network Percent error")
    # plt.xlabel("Epoch")
    # plt.ylabel("Error Percent")
    # plt.xlim(-2,epochs)
    # plt.ylim(-2,110)
    # plt.legend()

def part_2b():

    """
    **********************************************
    create training dataset
    **********************************************

    matlab dataset:

    dat1 = [normrnd(6,2,100,1) normrnd(2,2,100,1)];  % create the input data
    dat2 = [normrnd(2,3,100,1) normrnd(8,2,100,1)];
    dat = [dat1; dat2];

    y = [ones(100,1); 0*ones(100,1)];  % labels for classes
    """
    tempx = np.random.normal(6,2,size=100)
    tempy = np.random.normal(2,2,size=100)
    dat1 = np.array((tempx,tempy)).T

    tempx = np.random.normal(2,3,size=100)
    tempy = np.random.normal(8,2,size=100)
    dat2 = np.array((tempx,tempy)).T

    dat = np.append(dat1, dat2, axis=0) # [200x2]

    y = np.expand_dims(np.append(np.ones((100,1)),0*np.ones((100,1))),axis=1) # [200x1]

    """
    shuffle dataset
    """
    temp = np.append(dat,y,axis=1)
    np.random.shuffle(temp)
    dat = temp[:,:2]
    tempy = temp[:,-1]
    y = np.expand_dims(tempy,axis=1)

    """
    **********************************************
    create validation dataset
    **********************************************
    """
    v_tempx = np.random.normal(6,2,size=10)
    v_tempy = np.random.normal(2,1,size=10)
    v_dat1 = np.array((v_tempx,v_tempy)).T

    v_tempx = np.random.normal(2,3,size=10)
    v_tempy = np.random.normal(8,1,size=10)
    v_dat2 = np.array((v_tempx,v_tempy)).T

    v_dat = np.append(v_dat1, v_dat2, axis=0) # [200x2]

    v_y = np.expand_dims(np.append(np.ones((10,1)),0*np.ones((10,1))),axis=1) # [200x1]

    """
    shuffle dataset
    """
    v_temp = np.append(v_dat,v_y,axis=1)
    np.random.shuffle(v_temp)
    v_dat = v_temp[:,:2]
    v_tempy = v_temp[:,-1]
    v_y = np.expand_dims(v_tempy,axis=1)


    """
    **********************************************
    network parameters
    **********************************************
    """
    epochs = 50
    w = np.random.random((3,1)) # starting weight for each column (synapse)
    bias = np.ones((len(dat),1)) # bias
    error_arr = np.zeros((epochs,1))


    """
    **********************************************
    perceptron single layer network
    **********************************************
    """
    print "Weights Before Training: "
    print w
    print

    for i in range(epochs):
        """
        neuron
        """
        input = dat
        input = np.append(input,bias,axis=1)
        xw = np.dot(input,w) # [4x3]*[3*1]=[4x1]

        y_hat = sigmoid(xw)

        error = y - y_hat
        error_arr[i] = sum(error)

        adjustments = error * sigmoid_derivative(y_hat)

        w = w + np.dot(input.T,adjustments)
        # w = normalize(w)

    print "Weights After Training: "
    print w
    print

    print "Network Percent Error (%):"
    percent_error=(sum(y-np.round(y_hat,0))/epochs)*100
    print percent_error[0]
    print

    y_hat = np.dot(input,w)


    """
    **********************************************
    plotting
    **********************************************
    """
    fig7 = plt.figure()
    plt.plot(dat1.T[0], dat1.T[1], "ro", label="dat1")
    plt.plot(dat2.T[0], dat2.T[1], "bo", label="dat2")
    plt.title("Part B: 2 Classes, Initial Dataset")
    plt.xlim(-5,10)
    plt.ylim(-4,14)
    plt.legend()

    fig8 = plt.figure()
    plt.plot(dat1.T[0], dat1.T[1], "ro", label="dat1")
    plt.plot(dat2.T[0], dat2.T[1], "bo", label="dat2")
    # plt.plot((sum(y-np.round(y_hat,0))/epochs)*100,"wo",label="percent error: " + str(percent_error))# + str((sum(y-np.round(y_hat,0))/200)*100))
    for i in range(len(y_hat)):
    	if (y_hat[i]<0):
    		plt.plot(dat[i][0], dat[i][1],"b.")
    	else:
    		plt.plot(dat[i][0], dat[i][1],"r.")
    plt.title("Part 2b: 2 Classes, Sigmoidal Output")
    plt.xlim(-5,10)
    plt.ylim(-4,14)
    plt.legend()

    fig9 = plt.figure()
    plt.plot(abs(error_arr), label="Error")
    plt.title("Part 2b: Network Percent error")
    plt.xlabel("Epoch")
    plt.ylabel("Error Percent")
    plt.xlim(-2,epochs)
    plt.ylim(-2,110)
    plt.legend()

def part_2c():
    """
    **********************************************
    create training dataset
    **********************************************

    matlab dataset:

    dat1 = [normrnd(6,1,100,1) normrnd(2,1,100,1)];      % create the input data
    dat2 = [normrnd(2,1,100,1) normrnd(8,1,100,1)];
    dat3 = [normrnd(-2,1,100,1) normrnd(-2,1,100,1)];
    dat = [dat1; dat2; dat3];

    y(:,1) = [ones(100,1); zeros(100,1); zeros(100,1)];   % the class labels as three dimensional outputs
    y(:,2) = [zeros(100,1); ones(100,1); zeros(100,1)];
    y(:,3) = [zeros(100,1); zeros(100,1); ones(100,1)];
    """
    tempx = np.random.normal(6,1,size=100)
    tempy = np.random.normal(2,1,size=100)
    dat1 = np.array((tempx,tempy)).T

    tempx = np.random.normal(2,1,size=100)
    tempy = np.random.normal(8,1,size=100)
    dat2 = np.array((tempx,tempy)).T

    tempx = np.random.normal(-2,1,size=100)
    tempy = np.random.normal(-2,1,size=100)
    dat3 = np.array((tempx,tempy)).T

    dat = np.append(dat1, dat2, axis=0) # [300x2]
    dat = np.append(dat, dat3, axis=0)

    col1 = np.append(np.ones((100,1)),np.zeros((200,1)),axis=0)
    col2 = np.append(np.zeros((100,1)),np.ones((100,1)),axis=0)
    col2 = np.append(col2,np.zeros((100,1)),axis=0)
    col3 = np.append(np.zeros((200,1)),np.ones((100,1)),axis=0)

    y = np.append(col1,col2,axis=1)
    y = np.append(y,col3,axis=1)

    """
    shuffle dataset
    """
    temp = np.append(dat,y,axis=1)
    np.random.shuffle(temp)
    dat = temp[:,:2]
    tempy = temp[:,-3:]

    """
    **********************************************
    create validation dataset
    **********************************************
    """
    v_tempx = np.random.normal(6,2,size=10)
    v_tempy = np.random.normal(2,1,size=10)
    v_dat1 = np.array((v_tempx,v_tempy)).T

    v_tempx = np.random.normal(2,3,size=10)
    v_tempy = np.random.normal(8,1,size=10)
    v_dat2 = np.array((v_tempx,v_tempy)).T

    v_tempx = np.random.normal(-2,1,size=10)
    v_tempy = np.random.normal(-2,1,size=10)
    v_dat3 = np.array((v_tempx,v_tempy)).T

    v_dat = np.append(v_dat1, v_dat2, axis=0) # [300x2]
    v_dat = np.append(v_dat, v_dat3, axis=0)

    v_col1 = np.append(np.ones((10,1)),np.zeros((20,1)),axis=0)
    v_col2 = np.append(np.zeros((10,1)),np.ones((10,1)),axis=0)
    v_col2 = np.append(v_col2,np.zeros((10,1)),axis=0)
    v_col3 = np.append(np.zeros((20,1)),np.ones((10,1)),axis=0)

    v_y = np.append(v_col1,v_col2,axis=1)
    v_y = np.append(v_y,v_col3,axis=1)

    """
    shuffle dataset
    """
    v_temp = np.append(v_dat,v_y,axis=1)
    np.random.shuffle(v_temp)
    v_dat = v_temp[:,:2]
    v_tempy = v_temp[:,-3:]

    """
    **********************************************
    set network parameters
    **********************************************
    """
    epochs = 3000
    w = np.random.random((3,3)) # starting weight for each column (synapse)
    bias = np.ones((len(dat),1)) # bias
    error_arr = np.zeros((epochs,3))
    max=0

    print "Starting Weights: "
    print w
    print


    """
    **********************************************
    perceptron single layer network
    **********************************************
    """
    for i in range(epochs):
        """
        neuron
        """
        input = dat
        input = np.append(input,bias,axis=1)
        xw = np.dot(input,w)

        y_hat = sigmoid(xw)

        error = y - y_hat
        # print y.shape
        error_arr[i] = sum(error)
        adjustments = error * sigmoid_derivative(y_hat)

        w = w + np.dot(input.T,adjustments)
        # w = normalize(w)

    print "Weights after training: "
    print w
    print

    # print "percent error:"
    # percent_error=(sum(y-np.round(y_hat,0))/epochs)*100
    # print percent_error[0]
    # print

    y_hat = np.dot(input,w)

    for i in range(y_hat.shape[0]):

        max=y_hat[i][0]

        for j in range(y_hat.shape[1]):

            if (y_hat[i][j]>max):
                max = y_hat[i][j]

        for j in range(y_hat.shape[1]):

            if (y_hat[i][j]==max):
                y_hat[i][j] = 1
            else:
                y_hat[i][j] = 0

    # error_total = 0
    # percent_err = np.round((y_hat - y),0)
    # for i in range(len(percent_err)):
    #     if sum(percent_err[i]>0):
    #         error_total += sum(percent_err[i])
    #
    # print "Final Network Error: " + str(np.round((error_total/len(y))*100,2))


    """
    **********************************************
    plotting
    **********************************************
    """
    # print y_hat

    fig10 = plt.figure()
    plt.plot(dat1.T[0], dat1.T[1], "ro", label="dat1")
    plt.plot(dat2.T[0], dat2.T[1], "bo", label="dat2")
    plt.plot(dat3.T[0], dat3.T[1], "go", label="dat3")
    plt.title("Part 2c: 3 Classes, Initial Dataset")
    plt.xlim(-5,10)
    plt.ylim(-6,12)
    plt.legend()

    fig11 = plt.figure()
    plt.plot(dat1.T[0], dat1.T[1], "ro", label="dat1 predicted")
    plt.plot(dat2.T[0], dat2.T[1], "bo", label="dat2 predicted")
    plt.plot(dat3.T[0], dat3.T[1], "go", label="dat3 predicted")

    for i in range(len(y_hat)):
        if (y_hat[i][1]==1):
            plt.plot(dat[i][0], dat[i][1],"r.")
        elif (y_hat[i][2]==1):
            plt.plot(dat[i][0], dat[i][1],"b.")
        elif (y_hat[i][0]==1):
            plt.plot(dat[i][0], dat[i][1],"g.")

    plt.title("Part 2c: 3 Classes, Sigmoidal Output")
    plt.xlim(-5,10)
    plt.ylim(-6,12)
    plt.legend()

    y1_error = error_arr[:,0]
    y2_error = error_arr[:,1]
    y3_error = error_arr[:,2]

    y1_percent = np.empty([3000,1])
    y2_percent = np.empty([3000,1])
    y3_percent = np.empty([3000,1])

    for i in range(epochs):
        y1_percent[i] = sum(y1_error)/epochs
        y2_percent[i] = sum(y2_error)/epochs
        y3_percent[i] = sum(y3_error)/epochs

    # fig12 = plt.figure()
    # plt.plot(abs(y1_percent),"r",label="dat1 error")
    # plt.plot(abs(y2_percent),"b",label="dat2 error")
    # plt.plot(abs(y3_percent),"g",label="dat3 error")
    #
    # plt.title("Part 2c: Network Percent error")
    # plt.xlabel("Epoch")
    # plt.ylabel("Error")
    # plt.xlim(-2,epochs)
    # plt.ylim(-2,110)
    # plt.legend()

def main():

    """
    part 1
    """
    x, y = part_1a()
    h = part_1b(x)
    w = part_1c(y, h)
    part_1d(x, y, h, w)
    x, y, six_x, six_y= part_1e(x, y)
    f_x = part_1f(x, y, six_x, six_y)
    part_1g(x, f_x)

    """
    part 2
    """
    part_2a()
    part_2b()
    part_2c()
    plt.show()

if __name__ == '__main__':
    main()
