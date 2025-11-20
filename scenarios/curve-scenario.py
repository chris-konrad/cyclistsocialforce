# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 14:32:22 2025

@author: Christoph M. Konrad
"""

import matplotlib.pyplot as plt
import numpy as np

from scipy import interpolate
from cyclistsocialforce.scenario import Scenario
from cyclistsocialforce.vehicle import BalancingRiderBicycle
from cyclistsocialforce.parameters import RoadElementParameters
from cyclistsocialforce.intersection import (
    RoadEdge,
    StraightRoadSegment,
    CurvedRoadSegment,
    RoadSegmentCollection,
    SocialForceIntersection,
)

# %% Create a custom class that defines this scenario.

class CurveScenario(Scenario):
    """ A custom scenario of a cyclist following a curve.
    """
    
    def __init__(self, **kwargs):
        
        # set up a figure
        self._make_figure()
    
        # create road
        self.segs = self._make_road()
    
        # create intersection object to manage the road segments
        self.ins = SocialForceIntersection([], road_elements=[self.segs], 
                                           animate=True, axes=self.ax)
    
        # create a bike and att ot the intersection.
        b = BalancingRiderBicycle(
            (0, -5, np.pi / 2, 5, 0, 0), vid="BalancingRiderBike", saveForces=True
        )
        b.params.v_desired_default = 4.0
        destx, desty = self.segs.get_destinations_from_segments()
        for i in range(3):
            destx.append(destx[-1])
            desty.append(desty[-1]+1)
        b.setDestinations(destx, desty)
        b.add_drawing(self.ax)
        self.ins.add_road_user(b)
        
        # call super constructor
        kwargs['axes'] = self.ax
        super().__init__(self._step_func, **kwargs)
        
    def _make_figure(self):
        self.fig, self.ax = plt.subplots(1,1)
        self.ax.set_xlim(-5, 25)
        self.ax.set_ylim(-5, 45)
        
    def _make_road(self):
        
        # create road segments
        roadwidth = 5
        ds = 0.1
        x0 = np.array((0, -20, np.pi / 2))
        roadparams = RoadElementParameters(sigma=2.0, F_0=0.15)
        seg1 = StraightRoadSegment(x0, roadwidth, 25, params=roadparams, ds=ds)
        seg2 = CurvedRoadSegment(
            seg1.x1, roadwidth, 10, np.pi / 2, "right", params=roadparams, ds=ds
        )
        seg3 = CurvedRoadSegment(
            seg2.x1, roadwidth, 10, np.pi / 2, "left", params=roadparams, ds=ds
        )
        seg4 = StraightRoadSegment(seg3.x1, roadwidth, 20, params=roadparams, ds=ds)
    
        segs = RoadSegmentCollection((seg1, seg2, seg3, seg4))
        
        return segs
    
    def _step_func(self):
        """
        Step function for this scenario.
        """
        
        self.ins.step()
        
    def plot_force_field(self):
        
        # plot road force field
        x = np.arange(-5, 10, 0.1)
        y = np.arange(0, 25, 0.1)

        X, Y = np.meshgrid(x, y)
        Fx, Fy = self.segs.calcRepulsiveForce(X, Y)

        fig2, ax2 = plt.subplots(1, 2)
        ax2[0].set_aspect("equal")
        F = np.sqrt(Fx**2 + Fy**2)
        F[F > 5] = 5
        ax2[0].contourf(X, Y, F)


        x = np.arange(-5, 10, 1.0)
        y = np.arange(0, 25, 1.0)
        X, Y = np.meshgrid(x, y)
        Fx, Fy = self.segs.calcRepulsiveForce(X, Y)
        ax2[0].quiver(X, Y, Fx, Fy, color="white")


        ax2[0].set_xlim(-5, 10)
        ax2[0].set_ylim(0, 25)

        x = np.arange(-5, 10, 0.1)
        y = np.zeros_like(x)
        Fx, Fy = self.segs.calcRepulsiveForce(x, y)
        F = np.sqrt(Fx**2 + Fy**2)
        F[F > 10] = 10
        ax2[1].plot(x, F)

        fig3, ax3 = plt.subplots(6, 1, sharex=True)
        self.ins.vehicles[0].plot_states(ax3[0:6])
        self.ins.vehicles[0].plot_forces((ax3[2],), ("direction",))
    
def main():
    t_end = 15
    scn = CurveScenario(animate=True)
    scn.run(t_end)
    scn.ins.set_animated(False)
    scn.plot_force_field()

if __name__=="__main__":
    main()
    