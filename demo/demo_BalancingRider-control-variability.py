# -*- coding: utf-8 -*-
"""
BalancingRider Control Variability Demo

Demonstrate the path variability resulting from the distribution
of control parameters with the BalancingRiderModel.

Simulates 10 riders with different control parameters following the same 
sequence of commands. Riders do not react to each other.

@author: Christoph M. Konrad
"""

import matplotlib.pyplot as plt
import numpy as np
import argparse

from cyclistsocialforce.scenario import Scenario
from cyclistsocialforce.vehicle import BalancingRiderBicycle
from cyclistsocialforce.parameters import BalancingRiderBicycleParameters

from pypaperutils.design import TUDcolors

colors = np.array(TUDcolors().colormap().colors)

class ControlVariabilityScenario(Scenario):
    """ A custom scenario of mulitiples cyclist with different 
    control parameters following the same parcours.
    """
    
    def __init__(self, n_bicycles, **kwargs):

        bike_spacing = 5
        
        destx = np.array([0, 3, 5, 3, 5, 5])
        desty = np.array([10, 20, 30, 40, 50, 100])

        # set up a figure
        self._make_figure(bike_spacing * n_bicycles + np.max(destx) + 5)
        
        self.bikes = []
        self.colors = np.tile(colors, (np.ceil(n_bicycles/len(colors)).astype(int),1))
    
        for i in range(n_bicycles):
            
            offset_x = i * bike_spacing

            # Per default, BalancingRiderBicycle uses the average control paramters. 
            # To sample parameters from the control parameter distribution, set stochastic_control_behavior=True.
            params_i = BalancingRiderBicycleParameters(stochastic_control_behavior=True)
            bike_i = BalancingRiderBicycle((offset_x, 0, np.pi/2, 5, 0, 0, 0, 0), id=f"b{i:02d}", params=params_i)
            bike_i.params.v_desired_default = 4.0

            bike_i.setDestinations(destx + offset_x, desty)

            self.bikes.append(bike_i)

        # call super constructor
        kwargs['axes'] = self.ax
        super().__init__(self._step_func, **kwargs)

    def _make_figure(self, x_max):
        self.fig, self.ax = plt.subplots(1,1, layout='constrained')
        self.fig.set_size_inches(15,10)
        self.ax.set_aspect("equal")
        self.ax.set_xlim(-5, x_max)
        self.ax.set_ylim(1, 50)
        self.ax.set_ylabel("y [m]")
        self.ax.set_xlabel("x [m]")
        self.ax.set_title("Test: BalancingRiderDrawing (2D)")
    
    def _step_func(self):
        """
        Step function for this scenario. In the first step, each cyclist is 
        drawn with a different color.
        """

        for i, b in enumerate(self.bikes):
            if b.drawing is None:
                b.add_drawing(self.ax, draw_name=False,
                              bike_color_frame=self.colors[i,:],
                              traj_line_color=self.colors[i,:])
            Fx, Fy = b.calcDestinationForce()
            b.step(Fx, Fy)

def parse_args():
    parser = argparse.ArgumentParser(
        prog="demo_BalancingRider-control-variability.py",
        description=("Demonstrate the path variability resulting from "
                     "the distribution of control parameters with the "
                     "BalancingRiderModel.")
    )
    return parser.parse_args()

def main():
    parse_args()

    t_end = 15

    scn = ControlVariabilityScenario(animate=True, n_bicycles=10)
    scn.run(t_end)

    axes = None
    for b in scn.bikes:
        b.drawing.set_animated(False)
        axes=b.plot_states(axes=axes)

    plt.show(block=True)

if __name__=="__main__":
    main()