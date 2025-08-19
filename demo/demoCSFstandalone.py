# -*- coding: utf-8 -*-
"""Run a cyclist social force demo without SUMO.

Simulates three encroaching cyclists. 

usage: demoCSFstandalone.py [-h] [-s] [-i]

optional arguments:
  -h, --help  show this help message and exit
  -s, --save  Save results to ./output/
  -i, --use_inv_pendulum_bike Use an inverted pendulum bicycle model instead 
                              of the simple 2D model.

Created on Tue Feb 14 18:26:19 2023
@author: Christoph Schmidt
"""
import numpy as np
import matplotlib.pyplot as plt
import argparse

from datetime import datetime
from time import sleep, time
from os import mkdir, path


from cyclistsocialforce.vehicle import Bicycle, InvPendulumBicycle
from cyclistsocialforce.intersection import SocialForceIntersection


def initAnimation():
    """Initialize the animation of the demo.

    Use blitting for speed-up.

    Returns
    -------
    fig : figure handle
        The figure the animation will be shown in
    fig_bg : image
        The background of the figure for blitting
    ax : axes object
        The axes the animation will be shown in

    """
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlim(0, 30)
    ax.set_ylim(-10, 20)
    ax.set_aspect("equal")
    figManager = plt.get_current_fig_manager()
    figManager.resize(1000, 500)
    plt.show(block=False)
    plt.pause(0.1)
    fig_bg = fig.canvas.copy_from_bbox(fig.bbox)
    fig.canvas.blit(fig.bbox)

    return fig, fig_bg, ax


def run(intersection, t, fig, fig_bg):
    """Run the simulation of the demo intersection"""

    t_s = intersection.vehicles[0].params.t_s

    i_max = t / t_s

    i = 0
    while i < i_max:
        t = time()

        fig.canvas.restore_region(fig_bg)

        intersection.step()

        fig.canvas.blit(fig.bbox)
        fig.canvas.flush_events()

        dt = time() - t
        if dt < t_s:
            sleep(t_s - dt)

        i = i + 1

    # End animation to prevent animated graphics from disappearing.
    intersection.endAnimation()


def parseArgs():
    parser = argparse.ArgumentParser(
        description="Run a cyclist social force " "demo without SUMO."
    )
    parser.add_argument(
        "-s",
        "--save",
        dest="save",
        default=False,
        action="store_true",
        help="Save results to ./output/",
    )
    parser.add_argument(
        "-i",
        "--use_inv_pendulum_bike",
        dest="use_inv_pendulum_bike",
        default=False,
        action="store_true",
        help=(
            "Use an inverted pendulum bicycle model instead of the simple"
            "simple 2D model"
        ),
    )
    return parser.parse_args()


def main():
    """Run a cyclist social force demo without SUMO.

    This script:
        - creates some bicycles
        - gives them a destination
        - simulates their movement to the destinations
        - optionally shows:
            - force fields
            - potentials
            - force magnitude time histories
    """
    args = parseArgs()

    if args.save:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = path.join(
            "output", timestamp + "_testCyclistSocialForce.png"
        )
    else:
        filename = None

    plt.rcParams["text.usetex"] = False

    # Initialize the animation. If the simulation is run with SUMO, the
    # Scenario class takes care of that. Whithout Scenario, we have to do
    # it manually. TODO: Generalize Scenario class.
    fig, fig_bg, ax = initAnimation()

    # Create bicycle objects
    if args.use_inv_pendulum_bike:
        bike1 = InvPendulumBicycle(
            (-23 + 17, 0, 0, 5, 0, 0), id="a", saveForces=True
        )
        bike1.params.v_desired_default = 4.5
        print(bike1.params.v_desired_default)
        bike2 = InvPendulumBicycle(
            (0 + 15, -20, np.pi / 2, 5, 0, 0), id="b", saveForces=True
        )
        bike2.params.v_desired_default = 5.0
        bike3 = InvPendulumBicycle(
            (-2 + 15, -20, np.pi / 2, 5, 0, 0), id="c", saveForces=True
        )
        bike3.params.v_desired_default = 5.0
    else:
        bike1 = Bicycle((-23 + 17, 0, 0, 5, 0), id="a", saveForces=True)
        bike1.params.v_desired_default = 4.5
        bike2 = Bicycle(
            (0 + 15, -20, np.pi / 2, 5, 0), id="b", saveForces=True
        )
        bike2.params.v_desired_default = 5.0
        bike3 = Bicycle(
            (-2 + 15, -20, np.pi / 2, 5, 0), id="c", saveForces=True
        )
        bike3.params.v_desired_default = 5.0

    # Set destinations.
    bike1.setDestinations((35, 64, 65), (0, 0, 0))
    bike2.setDestinations((15, 15, 15), (20, 49, 50))
    bike3.setDestinations((13, 13, 13), (20, 49, 50))

    # A social force intersection to manage the three bicycles. Run the
    # simulation without SUMO. Activate animation for some nice graphics.
    intersection = SocialForceIntersection(
        (bike1, bike2, bike3),
        activate_sumo_cosimulation=False,
        animate=True,
        axes=ax,
    )

    # Run the simulation
    t = 7
    run(intersection, t, fig, fig_bg)

    axes = bike1.plot_states(t_end=t)
    bike2.plot_states(t_end=t, axes=axes)
    bike3.plot_states(t_end=t, axes=axes)

    if args.save:
        fig = axes[0].get_figure()
        fig.savefig(filename)


# Entry point
if __name__ == "__main__":
    main()
