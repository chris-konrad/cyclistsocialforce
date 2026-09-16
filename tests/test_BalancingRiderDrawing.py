# -*- coding: utf-8 -*-
"""
Test the BalancingRider drawing (2D).

This script runs a minimal animated scenario with a BalancingRider bicycle
following a predefined set of destinations.

Intended as a visual sanity test of the 2D visualization of roll/steer with
BalancingRiderDrawing.

Usage: 
    python test_BalancingRiderDrawing.py

@author: Christoph M. Konrad
"""


import matplotlib.pyplot as plt
import numpy as np
import argparse

from cyclistsocialforce.scenario import Scenario
from cyclistsocialforce.vehicle import BalancingRiderBicycle
from cyclistsocialforce.vizualisation import TUDcolors


def _drawingkwargs():
    return dict(draw_force_resulting=False, draw_force_destination=False,draw_forces_repulsive=False,draw_trajectory=False)

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
        destx = [0, 5, 7, 5, 10, 20, 21, 22, 23]
        desty = [10, 20, 30, 40, 50, 60, 70, 80, 90]
        
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
            self.bike.add_drawing(self.ax, dest_marker_color_cur='red', rider_color_body=TUDcolors().get("paars"), draw_force_resulting=False, draw_name=False)
        Fx, Fy = self.bike.calcDestinationForce()
        self.bike.step(Fx, Fy)
        self.ax.set_xlim(self.bike.s[0] + self.xrange)
        self.ax.set_ylim(self.bike.s[1] + self.yrange)

def plot_top():

    fig, axes = plt.subplots(3,1, layout='constrained', sharex=True, sharey=True)
    fig.set_size_inches(16,10)

    phi = np.deg2rad([-20, 0, 20])
    delta = np.deg2rad([20, 0, -20])

    drawing_kwargs = _drawingkwargs()

    k = 0
    for i, ax in enumerate(axes):
        ax.set_aspect("equal")
        ax.set_ylim(-2, 2)
        ax.set_xlim(-3, 15)
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.grid(True)

        n = 8
        psi = 2 * np.arange(n) * np.pi / n - np.pi
        x = 2 * np.arange(n)

        for j in range(n):
            bike = BalancingRiderBicycle((x[j], 0, psi[j], 5, delta[i], phi[i], 0, 0), id=f"BalancingRiderBike_{k}")   
            bike.add_drawing(ax, **drawing_kwargs)
            k+=1

    return fig, ax

def plot_side():
    fig, axes = plt.subplots(1,2, layout='constrained')
    fig.set_size_inches(12,6)

    phi = [-np.pi/2, np.pi/2]
    drawing_kwargs = _drawingkwargs()

    for i in range(2):

        ax= axes[i]
        ax.set_aspect("equal")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("z [m]")
        ax.set_ylim(-2, 2)
        ax.set_xlim(-1, 2)
        ax.grid(True)

        bike = BalancingRiderBicycle((0, 0, 0, 5, 0, phi[i], 0, 0), id=f"BalancingRiderBike_{i}")   
        bike.add_drawing(ax, **drawing_kwargs)

    return fig, axes

def parse_args():
    parser = argparse.ArgumentParser(
        prog="test_BalancingRiderDrawing.py",
        description="Test the 2D drawing of a BalancingRider bicycle following a parcours.",
    )

    parser.add_argument("--snapshots", nargs="+", type=float, help="Provide a list of times to save some snapshots.")

    return parser.parse_args()

def main():
    parse_args()

    t_end = 20

    scn = ParcoursScenario(animate=True)
    scn.run(t_end)
    scn.bike.drawing.set_animated(False)
    scn.bike.plot_states()

    plot_side()
    plot_top()

    plt.show(block=True)

if __name__=="__main__":
    main()