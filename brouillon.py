# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0,1,100)
y = np.linspace(0,1,100)

dx = x[1]
dy = y[1]

X,Y = np.meshgrid(x,y)

m = 10
n = 10

phi = np.sin(m * np.pi * X) * np.sin(n *np.pi * Y)

print(np.sum(phi**2) * dx * dy)

plt.figure()
plt.pcolor(X,Y,phi)