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
    MATLAB Dataset
    """
    tempx = np.random.normal(6,2,size=100)
    tempy = np.random.normal(2,1,size=100)
    dat1 = np.array((tempx,tempy)).T

    tempx = np.random.normal(2,3,size=100)
    tempy = np.random.normal(8,1,size=100)
    dat2 = np.array((tempx,tempy)).T

    dat = np.append(dat1, dat2, axis=0) # [200x2]

    y = np.expand_dims(np.append(np.ones((100,1)),-1*np.ones((100,1))),axis=1) # [200x1]

    temp = np.append(dat,y,axis=1)
    np.random.shuffle(temp)
    dat = temp[:,:2]
    tempy = temp[:,-1]
    y = np.expand_dims(tempy,axis=1)

    epochs = 3000
    weights = np.random.random((3,1)) # starting weight for each column (synapse)
    training_output = y
    bias = np.ones((len(dat),1)) # bias
    error_arr = np.zeros((epochs,1))

    print("weights before: ")
    print weights
    print
    # bias = 1 # bias
    bias = np.ones((len(dat),1)) # bias

    for i in range(epochs):
        """
        neuron
        """
        input = dat
        input = np.append(input,bias,axis=1)
        xw = np.dot(input,weights) # [4x3]*[3*1]=[4x1]

        output = sign(xw)

        error = training_output - output
        error_arr[i] = sum(error)

        adjustments = error * sign_derivative(output)

        weights = weights + np.dot(input.T,adjustments)
        # weights = normalize(weights)

    print "Weights after training: "
    print weights
    print

    print "percent error:"
    percent_error=(sum(y-np.round(output,0))/epochs)*100
    print percent_error[0]
    print

    y_hat = np.dot(input,weights)

    fig1 = plt.figure()
    plt.plot(dat1.T[0], dat1.T[1], "ro", label="dat1")
    plt.plot(dat2.T[0], dat2.T[1], "bo", label="dat2")
    # plt.plot(np.cross(weights.T,input))
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

def part_2b():
    """
    MATLAB Dataset
    """
    tempx = np.random.normal(6,2,size=100)
    tempy = np.random.normal(2,1,size=100)
    dat1 = np.array((tempx,tempy)).T

    tempx = np.random.normal(2,3,size=100)
    tempy = np.random.normal(8,1,size=100)
    dat2 = np.array((tempx,tempy)).T

    dat = np.append(dat1, dat2, axis=0) # [200x2]

    y = np.expand_dims(np.append(np.ones((100,1)),0*np.ones((100,1))),axis=1) # [200x1]

    temp = np.append(dat,y,axis=1)
    np.random.shuffle(temp)
    dat = temp[:,:2]
    tempy = temp[:,-1]
    y = np.expand_dims(tempy,axis=1)

    epochs=30
    weights = np.random.random((3,1)) # starting weight for each column (synapse)
    training_output = y
    bias = np.ones((len(dat),1)) # bias
    error_arr = np.zeros((epochs,1))

    print "Starting Weights: "
    print weights
    print

    for i in range(epochs):
        """
        neuron
        """
        input = dat
        input = np.append(input,bias,axis=1)
        xw = np.dot(input,weights) # [4x3]*[3*1]=[4x1]

        output = sigmoid(xw)

        error = training_output - output
        error_arr[i] = sum(error)

        adjustments = error * sigmoid_derivative(output)

        weights = weights + np.dot(input.T,adjustments)
        # weights = normalize(weights)

    print "Weights after training: "
    print weights
    print

    print "percent error:"
    percent_error=(sum(y-np.round(output,0))/epochs)*100
    print percent_error[0]
    print

    y_hat = np.dot(input,weights)

    fig1 = plt.figure()
    plt.plot(dat1.T[0], dat1.T[1], "ro", label="dat1")
    plt.plot(dat2.T[0], dat2.T[1], "bo", label="dat2")
    # plt.plot(np.cross(weights.T,input))
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

    """
    y(:,1) = [ones(100,1); zeros(100,1); zeros(100,1)];   % the class labels as three dimensional outputs
    y(:,2) = [zeros(100,1); ones(100,1); zeros(100,1)]';
    y(:,3) = [zeros(100,1); zeros(100,1); ones(100,1)]';
    """
    col1 = np.append(np.ones((100,1)),np.zeros((200,1)),axis=0)
    col2 = np.append(np.zeros((100,1)),np.ones((100,1)),axis=0)
    col2 = np.append(col2,np.zeros((100,1)),axis=0)
    col3 = np.append(np.zeros((200,1)),np.ones((100,1)),axis=0)

    y = np.append(col1,col2,axis=1)
    y = np.append(y,col3,axis=1)

    # temp = np.append(dat,y,axis=1)
    # np.random.shuffle(temp)
    # dat = temp[:,:2]
    # tempy = temp[:,-1]
    # y = np.expand_dims(tempy,axis=1)

    print y
    epochs=300
    weights = np.random.random((3,3)) # starting weight for each column (synapse)
    training_output = y
    bias = np.ones((len(dat),1)) # bias
    error_arr = np.zeros((epochs,1))

    print "Starting Weights: "
    print weights
    print

    for i in range(epochs):
        """
        neuron
        """
        input = dat
        input = np.append(input,bias,axis=1)
        xw = np.dot(input,weights)

        output = sigmoid(xw)

        error = training_output - output
        # error_arr[i] = sum(error)

        adjustments = error * sigmoid_derivative(output)

        weights = weights + np.dot(input.T,adjustments)
        # weights = normalize(weights)

    print "Weights after training: "
    print weights
    print

    print "percent error:"
    percent_error=(sum(y-np.round(output,0))/epochs)*100
    print percent_error[0]
    print

    y_hat = np.dot(input,weights)

    for i in range(len(y_hat)):
        for j in range(3):
            if (y_hat[i][j] > 0):
                y_hat[i][j] = 1
            else:
                y_hat[i][j] = 0


    print y_hat

    fig1 = plt.figure()
    plt.plot(dat1.T[0], dat1.T[1], "ro", label="dat1")
    plt.plot(dat2.T[0], dat2.T[1], "bo", label="dat2")
    plt.plot(dat3.T[0], dat3.T[1], "go", label="dat3")
    plt.plot((sum(y-np.round(output,0))/epochs)*100,"wo",label="percent error: " + str(percent_error))# + str((sum(y-np.round(output,0))/200)*100))


    plt.title("Linear Classification: Part C")
    plt.xlim(-5,10)
    plt.ylim(-6,12)
    plt.legend()

def main():
    # part_2a()
    # part_2b()
    part_2c()
    plt.show()

if __name__ == '__main__':
    main()
