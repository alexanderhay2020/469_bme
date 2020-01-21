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
    "returns normalized array between -1 and 1"

    arr = arr / np.abs(arr).max(axis=0)

    return arr

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
    v_tempx = np.random.normal(6,2,size=10)
    v_tempy = np.random.normal(2,1,size=10)
    v_dat1 = np.array((v_tempx,v_tempy)).T

    v_tempx = np.random.normal(2,3,size=10)
    v_tempy = np.random.normal(8,1,size=10)
    v_dat2 = np.array((v_tempx,v_tempy)).T

    v_dat = np.append(v_dat1, v_dat2, axis=0) # [200x2]

    y = np.expand_dims(np.append(np.ones((10,1)),-1*np.ones((10,1))),axis=1) # [200x1]

    """
    shuffle dataset
    """
    temp = np.append(v_dat,y,axis=1)
    np.random.shuffle(temp)
    v_dat = temp[:,:2]
    v_tempy = temp[:,-1]
    v_y = np.expand_dims(v_tempy,axis=1)


    """
    **********************************************
    set network parameters
    **********************************************
    """
    epochs = 3000
    w = np.random.random((3,1)) # starting weight for each column (synapse)
    training_output = y
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

        output = sign(xw)

        error = training_output - output
        error_arr[i] = sum(error)

        adjustments = error * sign_derivative(output)

        w = w + np.dot(input.T,adjustments)
        # w = normalize(w)

    print "Weights after training: "
    print w
    print

    print "percent error:"
    percent_error=(sum(y-np.round(output,0))/epochs)*100
    print percent_error[0]
    print

    y_hat = np.dot(input,w)

    """
    **********************************************
    plotting
    **********************************************
    """
    fig1 = plt.figure()
    plt.plot(dat1.T[0], dat1.T[1], "ro", label="dat1")
    plt.plot(dat2.T[0], dat2.T[1], "bo", label="dat2")
    # plt.plot(np.cross(w.T,input))
    plt.plot((sum(y-np.round(output,0))/epochs)*100,"wo",label="percent error: " + str(percent_error))# + str((sum(y-np.round(output,0))/200)*100))
    for i in range(len(y_hat)):
    	if (y_hat[i]<0):
    		plt.plot(dat[i][0], dat[i][1],"b.")
    	else:
    		plt.plot(dat[i][0], dat[i][1],"r.")
    plt.title("Part A: Classification using sign(wx)")
    plt.xlim(-5,10)
    plt.ylim(-4,14)
    plt.legend()

    fig2 = plt.figure()
    plt.plot(abs(error_arr), label="Error")
    plt.title("Network Percent error")
    plt.xlabel("Epoch")
    plt.ylabel("Error Percent")
    plt.xlim(-2,epochs)
    plt.ylim(-2,110)
    plt.legend()

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
    tempy = np.random.normal(2,1,size=100)
    dat1 = np.array((tempx,tempy)).T

    tempx = np.random.normal(2,3,size=100)
    tempy = np.random.normal(8,1,size=100)
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
    # vtemp = np.append(v_dat,y,axis=1)
    # np.random.shuffle(temp)
    # v_dat = temp[:,:2]
    # v_tempy = temp[:,-1]
    # v_y = np.expand_dims(v_tempy,axis=1)


    """
    **********************************************
    network parameters
    **********************************************
    """
    epochs = 50
    w = np.random.random((3,1)) # starting weight for each column (synapse)
    training_output = y
    bias = np.ones((len(dat),1)) # bias
    error_arr = np.zeros((epochs,1))


    """
    **********************************************
    perceptron single layer network
    **********************************************
    """
    print "Starting weights: "
    print w
    print

    for i in range(epochs):
        """
        neuron
        """
        input = dat
        input = np.append(input,bias,axis=1)
        xw = np.dot(input,w) # [4x3]*[3*1]=[4x1]

        output = sigmoid(xw)

        error = training_output - output
        error_arr[i] = sum(error)

        adjustments = error * sigmoid_derivative(output)

        w = w + np.dot(input.T,adjustments)
        # w = normalize(w)

    print "Weights after training: "
    print w
    print

    print "percent error:"
    percent_error=(sum(y-np.round(output,0))/epochs)*100
    print percent_error[0]
    print

    y_hat = np.dot(input,w)


    """
    **********************************************
    plotting
    **********************************************
    """
    fig1 = plt.figure()
    plt.plot(dat1.T[0], dat1.T[1], "ro", label="dat1")
    plt.plot(dat2.T[0], dat2.T[1], "bo", label="dat2")
    # plt.plot(np.cross(w.T,input))
    plt.plot((sum(y-np.round(output,0))/epochs)*100,"wo",label="percent error: " + str(percent_error))# + str((sum(y-np.round(output,0))/200)*100))
    for i in range(len(y_hat)):
    	if (y_hat[i]<0):
    		plt.plot(dat[i][0], dat[i][1],"b.")
    	else:
    		plt.plot(dat[i][0], dat[i][1],"r.")
    plt.title("Linear Classification: Part B")
    plt.xlim(-5,10)
    plt.ylim(-4,14)
    plt.legend()

    fig2 = plt.figure()
    plt.plot(abs(error_arr), label="Error")
    plt.title("Network Percent error")
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
    tempx = np.random.normal(6,2,size=100)
    tempy = np.random.normal(2,1,size=100)
    dat1 = np.array((tempx,tempy)).T

    tempx = np.random.normal(2,3,size=100)
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

    print y

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
    # temp = np.append(dat,y,axis=1)
    # np.random.shuffle(temp)
    # dat = temp[:,:2]
    # tempy = temp[:,-1]
    # y = np.expand_dims(tempy,axis=1)

    # print y
    """
    **********************************************
    set network parameters
    **********************************************
    """
    epochs = 300
    w = np.random.random((3,3)) # starting weight for each column (synapse)
    training_output = y
    bias = np.ones((len(dat),1)) # bias
    error_arr = np.zeros((epochs,1))

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

        output = sigmoid(xw)

        error = training_output - output
        # print error.shape
        # error_arr[i] = sum(error[i])

        adjustments = error * sigmoid_derivative(output)

        w = w + np.dot(input.T,adjustments)
        # w = normalize(w)

    print "Weights after training: "
    print w
    print

    print "percent error:"
    percent_error=(sum(y-np.round(output,0))/epochs)*100
    print percent_error[0]
    print

    y_hat = np.dot(input,w)

    for i in range(len(y_hat)):
        for j in range(3):
            if (y_hat[i][j] > 0):
                y_hat[i][j] = 1
            else:
                y_hat[i][j] = 0


    """
    **********************************************
    plotting
    **********************************************
    """
    print y_hat

    fig1 = plt.figure()
    plt.plot(dat1.T[0], dat1.T[1], "ro", label="dat1")
    plt.plot(dat2.T[0], dat2.T[1], "bo", label="dat2")
    plt.plot(dat3.T[0], dat3.T[1], "go", label="dat3")
    plt.plot((sum(y-np.round(output,0))/epochs)*100,"wo",label="percent error: " + str(np.round(percent_error,2)))# + str((sum(y-np.round(output,0))/200)*100))

    for i in range(len(y_hat)):
        if (y_hat[i][0]==1):
            plt.plot(dat[i][0], dat[i][1],"r.")
        elif (y_hat[i][1]==1):
            plt.plot(dat[i][0], dat[i][1],"b.")
        elif (y_hat[i][2]==1):
            plt.plot(dat[i][0], dat[i][1],"g.")

    plt.title("Linear Classification: Part C")
    plt.xlim(-5,10)
    plt.ylim(-6,12)
    plt.legend()

    fig2 = plt.figure()
    plt.plot(abs(error_arr), label="Error")
    plt.title("Network Percent error")
    plt.xlabel("Epoch")
    plt.ylabel("Error Percent")
    # plt.xlim(-2,epochs)
    # plt.ylim(-2,110)
    plt.legend()

def main():
    # part_2a()
    # part_2b()
    part_2c()
    plt.show()

if __name__ == '__main__':
    main()
