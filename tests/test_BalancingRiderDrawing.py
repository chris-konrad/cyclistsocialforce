# -*- coding: utf-8 -*-
"""
Test the BalancingRider drawing (2D)

@author: Christoph M. Konrad
"""

import matplotlib.pyplot as plt
import numpy as np

from cyclistsocialforce.scenario import Scenario
from cyclistsocialforce.vehicle import BalancingRiderBicycle

class ParcoursScenario(Scenario):
    """ A custom scenario of a cyclist following a parcours.
    """
    
    def __init__(self, **kwargs):
        
        # set up a figure
        self._make_figure()
    
        # create a bike and att ot the intersection.
        self.bike = BalancingRiderBicycle(
            (0, 0, np.pi/2, 5, 0, 0, 0, 0), id="BalancingRiderBike", saveForces=True
        )
        self.bike.params.v_desired_default = 4.0
        destx = [0, 10, 0, 5, 10, 20, 21, 22, 23]
        desty = [10, 20, 30, 40, 40, 40, 40, 40, 40]
        
        self.bike.setDestinations(destx, desty)

        # call super constructor
        kwargs['axes'] = self.ax
        super().__init__(self._step_func, **kwargs)

    def _make_figure(self):
        self.xrange = np.array((-5, 5))
        self.yrange = np.array((-5, 5))
        self.fig, self.ax = plt.subplots(1,1, layout='constrained')
        self.fig.set_size_inches(9,10)
        self.ax.set_aspect("equal")
        self.ax.set_xlim(self.xrange)
        self.ax.set_ylim(self.yrange)
        self.ax.set_ylabel("y [m]")
        self.ax.set_xlabel("x [m]")
        self.ax.set_title("Test: BalancingRiderDrawing (2D)")
    
    def _step_func(self):
        """
        Step function for this scenario.
        """
        if self.bike.drawing is None:
            self.bike.add_drawing(self.ax, dest_marker_color_cur='red')
        Fx, Fy = self.bike.calcDestinationForce()
        self.bike.step(Fx, Fy)
        self.ax.set_xlim(self.bike.s[0] + self.xrange)
        self.ax.set_ylim(self.bike.s[1] + self.yrange)
    
def main():
    t_end = 15
    scn = ParcoursScenario(animate=True)
    scn.run(t_end)
    scn.bike.drawing.set_animated(False)

if __name__=="__main__":
    main()