import numpy as np

np.random.seed(1)

# weights = np.random.random((2,1)) # starting weights [3x2]

# print("weights before: ")
# print weights

# Make a prediction with weights
def predict(row, weights):
	activation = weights[0]
	for i in range(len(row)-1):
		activation += weights[i] * row[i]
	return 1.0 if activation >= 0.0 else -1.0

# Estimate Perceptron weights using stochastic gradient descent
def train_weights(train, l_rate, n_epoch):
    weights = [0.0 for i in range(len(train[0]))]
    print("weights before: ")
    print weights
    for epoch in range(n_epoch):
    	sum_error = 0.0
    	for row in train:
    		prediction = predict(row, weights)
    		error = row[-1] - prediction
    		sum_error += error**2
    		weights[0] = weights[0] + l_rate * error
    		for i in range(len(row)-1):
    			weights[i + 1] = weights[i + 1] + l_rate * error * row[i]
    	print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, l_rate, sum_error))
    return weights

# test predictions
tempx = np.random.normal(6,2,size=100)
tempy = np.random.normal(2,1,size=100)
dat1 = np.array((tempx,tempy)).T

tempx = np.random.normal(2,3,size=100)
tempy = np.random.normal(8,1,size=100)
dat2 = np.array((tempx,tempy)).T

dat = np.append(dat1, dat2, axis=0) # [200x2]

y = np.expand_dims(np.append(np.ones((100,1)),0*np.ones((100,1))),axis=1) # [200x1]

dataset = np.append(dat,y,axis=1)

# bias = np.ones((len(dat),1)) # bias

# dataset = np.append(dat,bias,axis=1)

# for row in dataset:
# 	prediction = predict(row, weights)
# 	# print("Expected=%d, Predicted=%d" % (row[-1], prediction))

l_rate = 0.1
n_epoch = 500
weights = train_weights(dataset, l_rate, n_epoch)

print("weights after: ")
print weights
