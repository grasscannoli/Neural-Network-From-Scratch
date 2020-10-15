import NeuralNetwork as nn 
import numpy as np
import glob
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

fl = glob.glob('img_in/*.npy')
data = None
x = None
y = []
cnt = 0
for f in fl:
    tmp = np.load(f)
    if x is None:
        x = tmp
    else:
        x = np.vstack((x, tmp))
    y.extend([cnt]*len(tmp))
    cnt += 1
y = np.array(y)

x = x.reshape((x.shape[0], x.shape[1]))
y = y.reshape((y.shape[0], 1))
shuf = np.arange(len(y))

np.random.shuffle(shuf)
x = x[shuf]
y = y[shuf]

# x = StandardScaler().fit_transform(x)

# pca = PCA(n_components=25)
# x = pca.fit_transform(x)

# print(x, np.max(x), np.min(x))

model = nn.Sequential([
    (512, 'Nothing'),
    (5, 'tanh'),
    (5, 'tanh'),
    (5, 'softmax')
])

model.compile(optimizer='adam', loss='cross entropy')

model.fit(x, y, x[1200:], y[1200:], 1000)

model.test(x[1200:], y[1200:])