# -*- coding: utf-8 -*-
"""Run a cyclist social force demo without SUMO.

Simulates three encroaching cyclists. 

usage: demoCSFstandalone.py [-h] [-s] [-m]

optional arguments:
  -h, --help  show this help message and exit
  -s, --save  Save results to ./output/
  -m, --model Select between 'whipplecarvallo' (default), 'particle', 'invpendulum', and 'planartwowheel'.

Created on Tue Feb 14 18:26:19 2023
@author: Christoph Konrad
"""
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

from datetime import datetime

from cyclistsocialforce.vehicle import Bicycle, ParticleBicycle, InvPendulumBicycle, WhippleCarvalloBicycle
from cyclistsocialforce.intersection import SocialForceIntersection
from cyclistsocialforce.scenario import Scenario


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run a cyclist social force demo without SUMO."
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
        "-m",
        "--model",
        default="whipplecarvallo",
        type=str,
        help=("Choose the dynamic model for the simulated bicycles. Can be any of 'whipplecarvallo' (default), 'invpendulum', or 'planartwowheel'."),
    )
    return parser.parse_args()


def get_bike_type(argstr):
    """ Returns the bike model type selected by the '--model' input argument."""

    MODEL_TYPES = {
        "whipplecarvallo": WhippleCarvalloBicycle,
        "particle": ParticleBicycle,
        "invpendulum": InvPendulumBicycle,
        "planartwowheel": Bicycle,
    }

    if not argstr in MODEL_TYPES:
        raise ValueError(f"'model' must be any of {list(MODEL_TYPES.keys())}. Instead it was '{argstr}'")
    
    return MODEL_TYPES[argstr]


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
    args = parse_args()

    if args.save:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(
            "output", timestamp + "_standalone-demo-csf"
        )
    else:
        filename = None

    
    bike_type = get_bike_type(args.model)

    # Create a DemoScenario object based on the default simulation scenario. 
    # - set up the scenario in the constructor
    # - define a custom step function for the scenario
    class DemoScenario(Scenario):

        def __init__(self):
                """ Creates 3 bicycles witin one shared space without limits.
                """
                
                # Create bicycle objects
                bike1 = bike_type(
                    (-23 + 17, 0, 0, 5, 0, 0, 0, 0), id="a", saveForces=True
                )
                bike1.params.v_desired_default = 4.5
                print(bike1.params.v_desired_default)
                bike2 = bike_type(
                    (0 + 15, -20, np.pi / 2, 5, 0, 0, 0, 0), id="b", saveForces=True
                )
                bike2.params.v_desired_default = 5.0
                bike3 = bike_type(
                    (-2 + 15, -20, np.pi / 2, 5, 0, 0, 0, 0), id="c", saveForces=True
                )
                bike3.params.v_desired_default = 5.0

                # Set destinations.
                bike1.setDestinations((35, 64, 65), (0, 0, 0))
                bike2.setDestinations((15, 15, 15), (20, 49, 50))
                bike3.setDestinations((13, 13, 13), (20, 49, 50))

                # Axes for animation
                fig, ax = plt.subplots(1, 1)
                ax.set_title(f"Interaction demo: {args.model}")
                ax.set_xlim(0, 30)
                ax.set_ylim(-10, 20)
                ax.set_aspect("equal")

                # A social force intersection to manage the three bicycles. Run the
                # simulation without SUMO. Activate animation for some nice graphics.
                self.intersection = SocialForceIntersection(
                    (bike1, bike2, bike3),
                    activate_sumo_cosimulation=False,
                    animate=True,
                    axes=ax
                )

                Scenario.__init__(self, self._step_func, animate=True, verbose=True, axes=ax)

        def _step_func(self):
            """ The custom step function for this scenario calls the step function of the only "intersection" in the scenario.
            """
            self.intersection.step()
        
    # Run the simulation
    t = 7
    scn = DemoScenario()
    scn.run(t)

    # Prevent drawings from disappearing after the animation finished.
    scn.intersection.set_animated(False)
    
    # Plot simulation results
    axes_states = None
    axes_forces = None
    for bike in scn.intersection.vehicles:
        axes_states = bike.plot_states(t_end=t, axes=axes_states)
        axes_forces = bike.plot_forces(t_end=t, axes=axes_forces, components_to_plot=['magnitude', 'direction'])
    plt.show(block=True)

    if args.save:
        fig_states = axes_states[0].get_figure()
        fig_states.savefig(filename+"_states.png")

        fig_forces = axes_states[0].get_figure()
        fig_forces.savefig(filename+"_forces.png")


# Entry point
if __name__ == "__main__":
    main()
