# coding=utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt

x = [2,4,6,8,9,10,13,14,16,17,18,20,21,24,25,27,29,30,32,34,35,38,39,40]
y = [0,0,0.04,0.15,0.3,0.9,1.8,2,3,3.2,3.5,3.6,3.7,3.4,3.5,3.3,2.8,1.6,0.9,0.6,0.32,0.29,0.15,0]

distance = np.array(x)
wind_speed = np.array(y)


ax = plt.figure()
plt.plot(distance, wind_speed, marker='o')

plt.xlim([0, 45])
plt.ylim([0, 4])
plt.xlabel("distance(cm)")
plt.ylabel("wind speed(m/s)")


plt.savefig("figure.pdf")

plt.show()

