# -*- coding: utf-8 -*-

"""
Test the cyclist social force model (CSFM) with SUMO using the TraCI interface.

This script:
    - loads a simple example network
    - generates random bicycle demand
    - creates a CSFM scenario
    - launches the TraCI interface
    - runs the demo

To run, execute this script and then hit play in the SUMO GUI.

@author: Christoph Schmidt
"""

import os
import argparse
import numpy as np

import cyclistsocialforce.config as cfg

# Uncomment this to use libsumo instead of TraCI. Warning: Simulation will run
# without GUI.
# cfg.sumo_use_libsumo = True

if cfg.sumo_use_libsumo:
    import libsumo as traci
else:
    import traci
import sumolib

from cyclistsocialforce.scenario import SUMOScenario

import matplotlib.pyplot as plt


def generateRoutes():
    """Generate a route file for the demo scenario.

    A route file with random bicycle demand on the six routes of a three-legged
    intersection. Generates demand for 60 seconds.

    """
    # Generate random demand for r routes with probability p of a bike beeing
    # inserted at a time step. Generates demand for a total of t seconds.
    t = 60
    r = 6
    p = 5 / 9

    rng = np.random.default_rng()
    demand = rng.binomial(1, p, size=(t, r))

    fname_routefile = os.path.join(".", "config", "demoCSFxSUMO.rou.xml")

    with open(fname_routefile, "w") as routefile:
        print(
            "<routes>\n"
            "    <!-- Bicycle Vehicle Type -->\n"
            '    <vType id="bike" vClass="bicycle"'
            ' jmIgnoreJunctionFoeProb="1" jmIgnoreFoeProb="1"/>\n'
            "    <!-- Routes -->\n"
            '    <route id="r0" edges="-E31 -E29" />\n'
            '    <route id="r1" edges="E29 E30" />\n'
            '    <route id="r2" edges="-E31 E30" />\n'
            '    <route id="r3" edges="-E30 -E29" />\n'
            '    <route id="r4" edges="-E30 E31" />\n'
            '    <route id="r5" edges="E29 E31" />\n'
            "    <!-- Bikes -->",
            file=routefile,
        )

        vid = 0
        for i in range(t):
            for j in range(r):
                if demand[i, j]:
                    print(
                        '    <vehicle id="b%i" type="bike" route="r%i" '
                        'depart="%i" />' % (vid, j, i),
                        file=routefile,
                    )
                vid += 1

        print("</routes>", file=routefile)


def main():
    """Run a demo of a CSFM-controlled SUMO intersection.

    This script:
        - loads a simple example network
        - generates random bicycle demand
        - creates a CSFM scenario
        - launches the TraCI interface
        - runs the demo

    Use the start and stop buttons of SUMO to control the simulation.
    """
    parser = argparse.ArgumentParser(
        description="Run a demo of a CSFM-"
        "controlled SUMO intersection. Use "
        "start and stop buttons of the SUMO-GUI "
        "to control the simulation."
    )
    parser.parse_args()

    assert "SUMO_HOME" in os.environ, (
        "SUMO_HOME environment variable not set"
        "! See https://sumo.dlr.de/docs/"
        "Basics/Basic_Computer_Skills.html#"
        "sumo_home to solve this issue."
    )

    if cfg.sumo_use_libsumo:
        sumoBinary = sumolib.checkBinary("sumo")
    else:
        sumoBinary = sumolib.checkBinary("sumo-gui")

    generateRoutes()
    
    # bicycle drawing styles
    bicycle_drawing_kwargs = {"traj_line_width": 5}

    # set animate=True to show an animation of CSFM parallel to SUMO
    demo = SUMOScenario(
        os.path.join(".", "config", "demoCSFxSUMO.net.xml"),
        bicycle_type="Bicycle",
        animate=True,
        bicycle_drawing_kwargs = bicycle_drawing_kwargs,
    )

    # use TraCI to execute SUMO with a time resultion of 0.01 s
    traci.start(
        [
            sumoBinary,
            "-c",
            os.path.join(".", "config", "demoCSFxSUMO.sumocfg"),
            "--step-length",
            "0.01",
        ]
    )

    demo.run(n_steps=10000)

    # Uncomment this to show runtime measurment
    # demo.plot_runtime_vs_nvec()


# Entry point
if __name__ == "__main__":
    main()
