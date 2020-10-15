import NeuralNetwork_1 as nn 
import numpy as np
import glob
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

wt_file = "gen_delta_image/weights_83_2.pkl"

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

x = StandardScaler().fit_transform(x)
print(x[0].sum(), x[0].max())

#pca = PCA(n_components=25)
#x = pca.fit_transform(x)

model = nn.Sequential([
    (512, 'Nothing'),
    (5, 'tanh'),
    (5, 'tanh'),
    (5, 'softmax')
])

model.loadWb(wt_file)
model.compile(optimizer='generalized delta', loss='cross entropy')

model.test(x, y)
