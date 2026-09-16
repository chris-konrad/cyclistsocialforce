# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 14:32:22 2025

@author: Christoph M. Konrad
"""

import matplotlib.pyplot as plt
import numpy as np


from cyclistsocialforce.scenario import Scenario
from cyclistsocialforce.vehicle import BalancingRiderBicycle
from cyclistsocialforce.intersection import (
    SocialForceIntersection,
)

# %% Create a custom class that defines this scenario.
class ParcoursScenario(Scenario):
    """ A custom scenario of a cyclist following a curve.
    """
    
    def __init__(self, **kwargs):
        
        # set up a figure
        self._make_figure()
    
        # create intersection object to manage the road segments
        self.ins = SocialForceIntersection([], animate=True, axes=self.ax)
    
        # create a bike and att ot the intersection.
        b = BalancingRiderBicycle(
            (0, 0, np.pi / 2, 5, 0, 0, 0, 0), id="BalancingRiderBike", saveForces=True
        )
        b.params.v_desired_default = 4.0
        destx = [0, 10, 0, 5, 10, 20, 21, 22, 23]
        desty = [10, 20, 30, 40, 40, 40, 40, 40, 40]
        
        b.setDestinations(destx, desty)
        b.add_drawing(self.ax, dest_marker_color_cur='red')
        self.ins.add_road_user(b)
        
        # call super constructor
        kwargs['axes'] = self.ax
        super().__init__(self._step_func, **kwargs)
        
    def _make_figure(self):
        self.fig, self.ax = plt.subplots(1,1)
        self.ax.set_xlim(-5, 25)
        self.ax.set_ylim(-5, 45)
        
    
    def _step_func(self):
        """
        Step function for this scenario.
        """
        self.ins.step()
    
def main():
    t_end = 15
    scn = ParcoursScenario(animate=True)
    scn.run(t_end)
    scn.ins.set_animated(False)

if __name__=="__main__":
    main()
    