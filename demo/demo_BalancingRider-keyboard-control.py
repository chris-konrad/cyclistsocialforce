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
import keyboard
import sys

from cyclistsocialforce.scenario import Scenario
from cyclistsocialforce.vehicle import BalancingRiderBicycle
from cyclistsocialforce.parameters import BalancingRiderBicycleParameters

from pypaperutils.design import TUDcolors

colors = np.array(TUDcolors().colormap().colors)

import keyboard

def get_lr_key():
    if keyboard.is_pressed("right"):
        return 1
    elif keyboard.is_pressed("left"):
        return -1
    else:
        return 0

def check_exit():
    if keyboard.is_pressed("esc"):
        print("Exiting...")
        sys.exit(0)

class KeyboardControlScenario(Scenario):
    """ A custom scenario for controlling a cyclist with the keyboard.
    """
    
    def __init__(self, commanded_yaw_rate=1, **kwargs):

        self._make_figure()
        
        params = BalancingRiderBicycleParameters(stochastic_control_behavior=True)
        bike = BalancingRiderBicycle((0, 0, np.pi/2, 5, 0, 0, 0, 0), id="", params=params)
        bike.params.v_desired_default = 4.0
        self.bike = bike

        # call super constructor
        kwargs['axes'] = self.ax
        super().__init__(self._step_func, **kwargs)

        self.psi_c = 0
        self.psi_c_rate = commanded_yaw_rate

    def _make_figure(self):
        self.fig, self.ax = plt.subplots(1,1, layout='constrained')
        self.fig.set_size_inches(9,9)
        self.ax.set_aspect("equal")
        self.ax.set_xlim(-10, 10)
        self.ax.set_ylim(-10, 10)
        self.ax.set_ylabel("y [m]")
        self.ax.set_xlabel("x [m]")
        self.ax.set_title("Demo: BalancingRiderDrawing Keyboard Control \n Use left/right keys to steer the cyclist.")

    def _update_fov(self):

        self.ax.set_xlim(self.bike.s[0]-10, self.bike.s[0]+10)
        self.ax.set_ylim(self.bike.s[1]-10, self.bike.s[1]+10)
    
    def _step_func(self):
        """
        Step function for this scenario. In the first step, each cyclist is 
        drawn with a different color.
        """
        check_exit()

        if self.bike.drawing is None:
            self.bike.add_drawing(self.ax, draw_name=False)

        self.psi_c += get_lr_key() * self.psi_c_rate * self.t_s
        
        Fx = self.bike.params.v_desired_default * np.sin(self.psi_c)
        Fy = self.bike.params.v_desired_default * np.cos(self.psi_c)

        self.bike.step(Fx, Fy)
        self._update_fov()

def parse_args():
    parser = argparse.ArgumentParser(
        prog="demo_BalancingRider-keyboard-control.py",
        description=("Control the BalancingRider model with the left/right "
                     "keys on your keyboard. ")
    )
    return parser.parse_args()

def main():
    parse_args()

    t_end = 300

    scn = KeyboardControlScenario(animate=True)
    scn.run(t_end)

    plt.show(block=True)

if __name__=="__main__":
    main()