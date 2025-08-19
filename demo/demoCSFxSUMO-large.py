# -*- coding: utf-8 -*-
"""
Test the cyclist social force model (CSFM) toghether with SUMO using the TraCI
interface. 

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

from cyclistsocialforce.scenario import Scenario


def generateRoutes():
    """Generate a route file for the demo scenario.

    A route file with random bicycle demand on the six routes of a three-legged
    intersection. Generates demand for 60 seconds.

    """

    # Generate random demand for r routes with probability p of a bike beeing
    # inserted at a time step. Generates demand for a total of t seconds.
    t = 60
    r = 18
    p = 4.0 / 9

    rng = np.random.default_rng()
    demand = rng.binomial(1, p, size=(t, r))

    fname_routefile = os.path.join(".", "config", "demoCSFxSUMO-large.rou.xml")

    with open(fname_routefile, "w") as routefile:
        print(
            "<routes>\n"
            "    <!-- Bicycle Vehicle Type -->\n"
            '    <vType id="bike" vClass="bicycle"'
            ' jmIgnoreJunctionFoeProb="1" jmIgnoreFoeProb="1"/>\n'
            "    <!-- Routes -->\n"
            '    <route id="r0" edges="-E1 E2" />\n'
            '    <route id="r1" edges="-E1 E0 E7 E30 E4" />\n'
            '    <route id="r2" edges="-E1 E0 E7 E30 -E3 -E0 E2" />\n'
            '    <route id="r3" edges="-E1 E0 E7 E30 -E3 -E0 E1" />\n'
            '    <route id="r4" edges="-E1 E0 E7 E30 -E3 -E0 -E31 -E29" />\n'
            '    <route id="r5" edges="-E1 E0 E3 E4" />\n'
            '    <route id="r6" edges="-E1 E0 E3 -E30 -E29" />\n'
            '    <route id="r7" edges="-E1 E0 E3 -E30 E31 E2" />\n'
            '    <route id="r8" edges="-E1 E0 E3 -E30 E31 E1" />\n'
            '    <route id="r9" edges="E29 E30 E4" />\n'
            '    <route id="r10" edges="E29 E30 -E3 E7 -E29" />\n'
            '    <route id="r11" edges="E29 E30 -E3 E7 E31 E2" />\n'
            '    <route id="r12" edges="-E4 -E30 -E29" />\n'
            '    <route id="r13" edges="-E4 -E3 -E0 E2" />\n'
            '    <route id="r14" edges="-E4 -E3 E7 E31 E1" />\n'
            '    <route id="r15" edges="-E2 -E31 E30 E4" />\n'
            '    <route id="r16" edges="-E2 E0 E3 E4" />\n'
            '    <route id="r17" edges="-E2 -E31 -E29" />\n'
            "    <!-- Bikes -->",
            file=routefile,
        )

        id = 0
        for i in range(t):
            for j in range(r):
                if demand[i, j]:
                    print(
                        '    <vehicle id="b%i" type="bike" route="r%i" '
                        'depart="%i" />' % (id, j, i),
                        file=routefile,
                    )
                id += 1

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

    # set animate=True to show an animation of CSFM parallel to SUMO
    demo = Scenario(
        os.path.join(".", "config", "demoCSFxSUMO-large.net.xml"),
        bicycle_type="Bicycle",
        animate=True,
        run_time_factor=0.3,
    )

    if 1:
        # use TraCI to execute SUMO with a time resultion of 0.01 s
        traci.start(
            [
                sumoBinary,
                "-c",
                os.path.join(".", "config", "demoCSFxSUMO-large.sumocfg"),
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
