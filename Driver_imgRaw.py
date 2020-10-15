import NeuralNetwork_1 as nn 
import numpy as np
import glob
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

fl = glob.glob('img_raw/*.npy')
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

actx = None
for i in range(x.shape[0]):
    if actx is None:
        actx = x[i].flatten()
    else:
        actx = np.vstack((actx, x[i].flatten()))

x = actx

shuf = np.arange(len(y))

np.random.shuffle(shuf)
x = x[shuf]
y = y[shuf]

x = StandardScaler().fit_transform(x)
print(x[0].sum(), x[0].max())

print(x.shape)

pca = PCA(n_components=150)
x = pca.fit_transform(x)

model = nn.Sequential([
    (150, 'Nothing'),
    (4, 'tanh'),
    (4, 'tanh'),
    (5, 'softmax')
])

model.compile(optimizer='adam', loss='cross entropy')

model.fit(x[:1200], y[:1200], x[1200:], y[1200:], 10000)