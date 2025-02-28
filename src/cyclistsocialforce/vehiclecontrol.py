# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 16:50:10 2023.

@author: Christoph Konrad
"""
from cyclistsocialforce.utils import angleDifference

class PIDcontroller:
    
    def __init__(self, kp, ki, kd, dT, isangle=False):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.dT = dT
        self.e = 0
        self.i = 0
        self.isangle=isangle
        self.hist = []
        
    def step(self, e):
        
        #differential component
        if 0:
            de = angleDifference(self.e, e)
        else:
            de = self.e - e
        self.d = self.kd * de/self.dT
        self.e = e
        
        #integral component
        self.i = self.i + (self.ki * self.e * self.dT)    
        
        #proportional component
        self.p = self.kp * self.e
        
        # output
        out = self.p + self.i + self.d
        
        self.hist.append(e)
        
        return out