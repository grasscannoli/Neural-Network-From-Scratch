import NeuralNetwork_1 as nn 
import numpy as np
import matplotlib.pyplot as plt
from mayavi import mlab
from matplotlib import cm

f = open("func_app_in/train100.txt",'r')
l = f.readlines()
f.close()

x = np.array([(float(s.split(' ')[0]), float(s.split(' ')[1])) for s in l])
y = np.array([float(s.split(' ')[2]) for s in l])
y = np.reshape(y, (np.shape(y)[0], 1))

wtfile = "reg_wts.pkl"

model = nn.Sequential([
    (2, 'Nothing'),
    (50, 'tanh'),
    (50, 'tanh'),     
    (1, 'linear')
])

model.compile(optimizer='generalized delta', loss='cross entropy')
model.loadWb(wtfile)

op = model.test_r(x,y)

print(op)
