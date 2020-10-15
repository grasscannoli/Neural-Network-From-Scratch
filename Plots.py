import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# r_p = mpatches.Patch(color = "red", label = "Delta Optimizer")
b_p = mpatches.Patch(color = "blue", label = "Generalized Delta Optimizer")
# y_p = mpatches.Patch(color = "yellow", label = "Adam Optimizer")

plt.legend(handles = [b_p])

# f = open("image/epoch_loss_delta_img_1.txt", 'r')
# l = f.readlines()
# x = np.array([(float(pt.split()[0]), float(pt.split()[1])) for pt in l])
# f.close()
# plt.plot(x[:, 0], x[:, 1], "r")

f = open("func_app/epoch_loss_img_2.txt", 'r')
l = f.readlines()
x = np.array([(float(pt.split()[0]), float(pt.split()[1])) for pt in l])
f.close()
print(x)
plt.plot(x[:, 0], x[:, 1], "b")

# f = open("image/epoch_loss_adam_img_1.txt", 'r')
# l = f.readlines()
# x = np.array([(float(pt.split()[0]), float(pt.split()[1])) for pt in l])
# f.close()
# plt.plot(x[:, 0], x[:, 1], "y")

# plt.scatter(x[:, 0], x[:, 1], c = y.reshape((1500,)).astype(int))
plt.show()