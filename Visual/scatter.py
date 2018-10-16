# coding=utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt

x = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5]
y1 = [20, 14, 11, 6, 4, 1, 1]
y2 = [18, 15, 14, 10, 9, 8, 7]

max_speed = np.array(x)
no_estimation = np.array(y1) / 20.0
estimation = np.array(y2) / 20.0


ax = plt.figure()
plt.scatter(max_speed, no_estimation, c='r')
plt.scatter(max_speed, estimation, c='b')

plt.legend(["no estimation", "estimation"])
plt.xlim([0, 1.75])
plt.ylim([0, 1])
plt.xlabel("max speed")
plt.ylabel("accuracy")


plt.savefig("figure.pdf")

plt.show()

