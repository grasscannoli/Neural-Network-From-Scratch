import NeuralNetwork_1 as nn 
import numpy as np
import matplotlib.pyplot as plt

f = open("func_app_in/train100.txt",'r')
l = f.readlines()[1:]
f.close()

x = np.array([(float(s.split(' ')[0]), float(s.split(' ')[1])) for s in l])
y = np.array([float(s.split(' ')[2]) for s in l])
y = np.reshape(y, (np.shape(y)[0], 1))

f = open("func_app_in/val.txt",'r')
l = f.readlines()[1:]
f.close()

x_val = np.array([(float(s.split(' ')[0]), float(s.split(' ')[1])) for s in l])
y_val = np.array([float(s.split(' ')[2]) for s in l])
y_val = np.reshape(y_val, (np.shape(y_val)[0], 1))

model = nn.Sequential([
    (2, 'Nothing'),
    (50, 'tanh'),
    (50, 'tanh'),     
    (1, 'linear')
])

# # model.loadWb("weights_2.pkl")
model.compile(optimizer='generalized delta', loss='cross entropy')

model.fit_r(x, y, 500)

op = model.test_r(x_val,y_val)
print(op.shape,y_val.shape)
plt.scatter(y_val,op)
plt.axis('equal')
plt.show()
print(model.W[-1])