import NeuralNetwork_1 as nn 
import numpy as np

f = open("traingroup1.csv",'r')
l = f.readlines()[1:]
f.close()
wt_file="delta_2d_non_linear/good_wts_adam.pkl"

x = np.array([(float(s.split(',')[0]), float(s.split(',')[1])) for s in l])
# x /= np.sum(x, axis = 0, keepdims = True)
y = np.array([float(s.split(',')[2]) for s in l])
y = np.reshape(y, (np.shape(y)[0], 1))

print(x.shape)
print(y.shape)

model = nn.Sequential([
    (2, 'Nothing'),
    (6, 'tanh'),
    (6, 'tanh'),     
    (3, 'softmax')
])

model.loadWb(wt_file)
model.compile(optimizer='generalized delta', loss='cross entropy')

model.test(x,y)
