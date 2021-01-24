from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import numpy.random as rd
import random as random

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
# X, Y, Z = axes3d.get_test_data(0.05)
theta = np.linspace(-2 * np.pi, 2 * np.pi, 100)
centers_X = np.arange(0, 360, 36)
centers_Y = np.arange(0, 720, 72)
center_X1 = 2400
center_Y1 = 0
r1 = 500

r = np.arange(2500, 3000, 50)
z = np.arange(5, 6, 0.1)
x2 = 2400
y2 = 0
X2 = []
Y2 = []
k = 0
for i in range(10):
    X = centers_X[i] + r[i] * np.cos(theta)
    Y = centers_Y[i] + r[i] * np.sin(theta)
    Z = z[i]
    X1 = center_X1 + r1 * np.cos(theta)
    Y1 = center_Y1 + r1 * np.sin(theta)
    x2 = x2 + rd.random()
    y2 = y2 + rd.random()
    X2.append(x2)
    Y2.append(y2)
    #ax.plot(X, Y, Z, color='orange', linewidth=1)
    #ax.plot(X1, Y1, Z, color='red', linewidth=1.5)
    #ax.plot(X, Y, color='orange', linewidth=1)
    #ax.plot(X1, Y1, color='red', linewidth=1.5)

for j in range(1000000):
    t = 2 * np.pi * random.random()

    r_random = np.sqrt(random.random())
    x2 = r[0] * r_random * np.cos(t)
    y2 = r[0] * r_random * np.sin(t)
    X2 = []
    Y2 = []
    for n in range(10):
        x2 = x2 + 60 * 6 * (0.1 + random.uniform(-0.05, 0.05))
        y2 = y2 + 60 * 6 * (0.2 + random.uniform(-0.05, 0.05))
        X2.append(x2)
        Y2.append(y2)
    #ax.plot(X2, Y2, z, color='blue', linewidth=0.1)
    #ax.plot(X2, Y2, color='blue', linewidth=0.1)
    dx, dy = np.array(X2) - center_X1, np.array(Y2) - center_Y1
    d = (dx ** 2 + dy ** 2) ** 0.5

    #print(min(d))
    if min(d) <= r1:
        k = k+1
print(k)
#ax.plot(centers_X, centers_Y, color='blue', linewidth=1)

#plt.show()

