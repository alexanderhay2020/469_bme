import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1)

tempx = np.random.normal(6,2,size=100)
tempy = np.random.normal(2,2,size=100)
dat1 = np.array((tempx,tempy)).T

tempx = np.random.normal(2,3,size=100)
tempy = np.random.normal(8,2,size=100)
dat2 = np.array((tempx,tempy)).T

dat = np.append(dat1, dat2, axis=0) # [200x2]

y = np.expand_dims(np.append(np.ones((100,1)),0*np.ones((100,1))),axis=1) # [200x1]

temp = np.append(dat,y,axis=1)
np.random.shuffle(temp)
dat = temp[:,:2]
tempy = temp[:,-1]
y = np.expand_dims(tempy,axis=1)

w = np.random.random((3,2)) # starting weights [3x2]
v = np.random.random((3,1))

# bias = 1 # bias
bias = np.ones((len(dat),1)) # bias


# feature_set = np.array([[0,1,0],[0,0,1],[1,0,0],[1,1,0],[1,1,1]])
# labels = np.array([[1,0,0,1,1]])
# labels = labels.reshape(5,1)
feature_set = dat
labels = y

def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_der(x):
    return sigmoid(x) *(1-sigmoid (x))

wh = np.random.rand(len(feature_set[0]),4)
wo = np.random.rand(4, 1)
lr = 0.5

for epoch in range(2000):
    # feedforward
    zh = np.dot(feature_set, wh)
    ah = sigmoid(zh)

    zo = np.dot(ah, wo)
    ao = sigmoid(zo)

    # Phase1 =======================

    error_out = ((1 / 2) * (np.power((ao - labels), 2)))
    print(error_out.sum())

    dcost_dao = ao - labels
    dao_dzo = sigmoid_der(zo)
    dzo_dwo = ah

    dcost_wo = np.dot(dzo_dwo.T, dcost_dao * dao_dzo)

    # Phase 2 =======================

    # dcost_w1 = dcost_dah * dah_dzh * dzh_dw1
    # dcost_dah = dcost_dzo * dzo_dah
    dcost_dzo = dcost_dao * dao_dzo
    dzo_dah = wo
    dcost_dah = np.dot(dcost_dzo , dzo_dah.T)
    dah_dzh = sigmoid_der(zh)
    dzh_dwh = feature_set
    dcost_wh = np.dot(dzh_dwh.T, dah_dzh * dcost_dah)

    # Update Weights ================

    wh -= lr * dcost_wh
    wo -= lr * dcost_wo
