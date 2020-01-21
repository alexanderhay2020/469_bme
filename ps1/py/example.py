import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x):
	return 1.0/(1.0 + np.exp(-x))

def sigmoid_der(x):
	return x*(1.0 - x)

class NN:
    def __init__(self, dat):
        self.dat = dat
        self.l=len(self.dat)
        self.li=len(self.dat[0])

        self.wi=np.random.random((self.li, self.l))
        self.wh=np.random.random((self.l, 1))

    def think(self, inp):
        s1=sigmoid(np.dot(inp, self.wi))
        s2=sigmoid(np.dot(s1, self.wh))
        return s2

    def train(self, dat,coutputs, it):
        for i in range(it):
            l0=dat
            l1=sigmoid(np.dot(l0, self.wi))
            l2=sigmoid(np.dot(l1, self.wh))

            l2_err=outputs - l2
            l2_delta = np.multiply(l2_err, sigmoid_der(l2))

            l1_err=np.dot(l2_delta, self.wh.T)
            l1_delta=np.multiply(l1_err, sigmoid_der(l1))

            self.wh+=np.dot(l1.T, l2_delta)
            self.wi+=np.dot(l0.T, l1_delta)

# dat=np.array([[0,0], [0,1], [1,0], [1,1] ])
# outputs=np.array([ [0], [1],[1],[0] ])

tempx = np.random.normal(6,2,size=100)
tempy = np.random.normal(2,2,size=100)
dat1 = np.array((tempx,tempy)).T

tempx = np.random.normal(2,3,size=100)
tempy = np.random.normal(8,2,size=100)
dat2 = np.array((tempx,tempy)).T

dat = np.append(dat1, dat2, axis=0)

outputs = np.expand_dims(np.append(np.ones((100,1)),0*np.ones((100,1))),axis=1)
# w = np.random.random((2,1)) # starting weights [2x1]
# v = np.random.random((1,1))

n=NN(dat)
# print(n.think(dat))
n.train(dat, outputs, 1000)
y_hat=(n.think(dat))

plt.plot(dat1.T[0], dat1.T[1], "r.", label="dat1")
plt.plot(dat2.T[0], dat2.T[1], "b.", label="dat2")
for i in range(len(outputs)):
	if (y_hat[i]==0):
		plt.plot(dat[i][0], dat[i][1],"bx")
	else:
		plt.plot(dat[i][0], dat[i][1],"rx")
plt.title("Linear Classification: Part B")
plt.xlim(-5,10)
plt.ylim(-4,14)
plt.legend()
plt.show()
