import numpy as np
import random

def nonlin(x,deriv=False):
	if(deriv==True):
		return x*(1-x)

	return 1/(1+np.exp(-x))

np.random.seed(1)

I = 0.1418*np.ones(20)                       # kgm^2
theta = 50*(np.pi/180)

w = np.random.normal(173, 12, 20)
w = w*(np.pi/180)                            # rad/sec

t = theta/w                                  # sec
torque = (I*w)/t                             # kgm^2/s^2

w = np.expand_dims(w,axis=1)
t = np.expand_dims(t,axis=1)
torque = np.expand_dims(torque,axis=1)

input = np.append(t,torque,axis=1)
output = w

# randomly initialize our weights with mean 0
w0 = 2*np.random.random((3,2)) - 1
w1 = 2*np.random.random((3,1)) - 1

# bias = np.array([[1,1]])

# w0=np.append(w0,np.array([[1,1]]),axis=0)
# w1=np.append(w1,np.array([[1]]),axis=0)

# w2 = 2*np.random.random((4,1)) - 1
# w3 = 2*np.random.random((3,1)) - 1

epochs = 10000


# for i in range(10):
for i in range(epochs):
	index = random.randint(0,len(input)-1)

	# Feed forward through layers 0, 1, and 2
	l0 = np.expand_dims(input[index],axis=0)
	l0 = np.append(l0,1)

	l1 = nonlin(np.dot(l0,w0))
	l1 = np.append(l1,1)

	l2 = nonlin(np.dot(l1,w1))
	# l3 = nonlin(np.dot(l2,w2))
	# l4 = nonlin(np.dot(l3,w3))

	# print "l0 shape: " + str(l0.shape)
	# print "l1 shape: " + str(l1.shape)
	# print "l2 shape: " + str(l2.shape)
	# print

	# how much did we miss the target value?
	l2_error = output[index] - l2

	if (i% 100) == 0:
		print "Epoch: " + str(i) + "/" + str(epochs)
		print "Error:" + str(np.mean(np.abs(l2_error)))
		print

	# in what direction is the target value?
	# were we really sure? if so, don't change too much.
	# l4_delta = l4_error*nonlin(l4,deriv=True)
	# l3_error = l4_delta.dot(w3.T)
	#
	# l3_delta = l3_error*nonlin(l3,deriv=True)
	# l2_error = l3_delta.dot(w2.T)

	l2_delta = l2_error*nonlin(l2,deriv=True)
	l1_error = l2_delta.dot(w1.T)

	l1_delta = l1_error * nonlin(l1,deriv=True)

	# w3 += l1.T.dot(l4_delta)
	# w2 += l1.T.dot(l3_delta)
	print "l2"
	print l2.shape
	print
	print "l2 error"
	print l2_error.shape
	print
	print "l2 delta"
	print l2_delta.shape
	print
	print "w1"
	print w1.shape
	print
	print "l1"
	print l1.shape
	print

	w1 += np.dot(l1.T,l2_delta)
	w0 += l0.T.dot(l1_delta)

l0 = input
l1 = nonlin(np.dot(l0,w0))
l2 = nonlin(np.dot(l1,w1))
# l3 = nonlin(np.dot(l2,w2))
# l4 = nonlin(np.dot(l3,w3))

print
print "velocity: "
print w
print

print "output: "
print l2
print
