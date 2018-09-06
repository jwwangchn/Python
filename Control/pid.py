# coding=utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

class PID:
    def __init__(self, Kp, Ki, Kd, max):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.preErr = 0.0
        self.sumErr = 0.0
        self.dErr = 0.0
        self.output = 0.0
        self.outMax = max
        self.stop = False

    def reset(self):
        self.preErr = 0.0
        self.sumErr = 0.0
        self.dErr = 0.0

    def calc(self, curErr):
        self.sumErr = self.sumErr + curErr
        self.dErr = curErr - self.preErr
        self.preErr = curErr
        Kisum = self.Ki * self.sumErr
        if Kisum > 0.8:
            Kisum = 0.8
        if Kisum < -0.8:
            Kisum = -0.8
        self.output = self.Kp * curErr + Kisum + self.Kd * self.dErr
        if abs(self.output) > self.outMax:
            self.output = self.outMax * self.output / abs(self.output)

        if self.stop == True:
            self.output = self.output / 5
        return self.output

if __name__ == "__main__":
    pid = PID(Kp=1, Ki=0, Kd=0, max=1)