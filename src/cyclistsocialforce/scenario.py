# -*- coding: utf-8 -*-
"""
Created on Thu Apr 20 10:19:21 2023.

@author: Christoph Schmidt
"""

import sys
import numpy as np
import traceback

from math import sqrt, floor, ceil

import cyclistsocialforce.config as cfg
from cyclistsocialforce.vehicle import Bicycle, TwoDBicycle, InvPendulumBicycle
from cyclistsocialforce.parameters import (
    BicycleParameters,
    InvPendulumBicycleParameters,
)
from cyclistsocialforce.intersection import SocialForceIntersection
from cyclistsocialforce.utils import angleSUMOtoSFM

import matplotlib.pyplot as plt

from time import time, sleep
from datetime import timedelta

try:
    import sumolib

    if cfg.sumo_use_libsumo:
        import libsumo as traci
    else:
        import traci
except ImportError:
    raise ImportError(
        (
            "SUMO packages not found. The scenario module is "
            "designed to run SUMO scenarios. Install sumolib and "
            "either traci or libsumo to run cyclistsocialforce with "
            "SUMO. If you intend to run cyclistsocialforce "
            "with SUMO, set up your own scenarios as demonstrated "
            "in the demos."
        )
    )


class Scenario:
    def __init__(
        self,
        step_func,
        t_0=0,
        t_s=0.01,
        t_r=0.01,
        animate=False,
        axes=None,
        verbose=True,
        t_snapshots=(),
    ):
        self.t = t_0
        self.t_s = t_s
        self.t_r = t_r
        self.t_0 = t_0

        self.t_wall = time()

        self.i = 0

        self.animate = animate
        self.ax = axes

        self.verbose = verbose

        self.step_func = step_func

    def run(self, t_end):
        if self.verbose:
            input("\nPress any key to start simulation ... \n")

        t_start = time()
        if self.animate:
            self.run_animated(t_start, t_end)
        else:
            self.run_silent(t_start, t_end)

        t_end = str(timedelta(seconds=time() - t_start))[:-3]
        if self.verbose:
            print("\n")
        if self.verbose:
            print(f"Simulation finished after {t_end}")

    def run_silent(self, t_start, t_end):
        i_end = int(t_end / self.t_s)
        len_prev_msg = 0

        while self.i < i_end:
            t = time()
            self.step()
            len_prev_msg = self.wait(t, t_start, i_end, len_prev_msg)

    def run_animated(self, t_start, t_end):
        self.init_animation()

        i_end = int(t_end / self.t_s)
        len_prev_msg = 0

        while self.i < i_end:
            t = time()
            self.step_blitting()
            len_prev_msg = self.wait(t, t_start, i_end, len_prev_msg)

    def step_blitting(self):
        self.fig.canvas.restore_region(self.fig_bg)

        self.step()

        self.fig.canvas.blit(self.fig.bbox)
        self.fig.canvas.flush_events()

    def step(self):
        self.step_func()

        self.i += 1
        self.t += self.t_s

    def wait(self, t, t_start, i_end, len_prev_msg):
        if self.verbose:
            print("\r", end="")

        sim_time = str(timedelta(seconds=self.t))[:11]
        wall_time = str(timedelta(seconds=(time() - t_start)))[:11]

        dt = time() - t
        t_sleep = max(0, self.t_r - dt)

        if self.verbose:
            msg = f"Running step {self.i}/{i_end}, Sim. time {sim_time}, Wall time {wall_time}, Wall freq. {int(1/(dt+t_sleep))} Hz "
            msg += " " * max(len_prev_msg - len(msg), 0)
            print(msg, end="")
        else:
            msg = ""

        if dt < self.t_r:
            sleep(t_sleep)

        return len(msg)
    
    def reset(self):
        self.i = 0
        self.t = self.t_0

    def init_animation(self):
        """Initialize the animation of the scenario.

        Uses blitting for faster animation.
        (https://matplotlib.org/stable/users/explain/animations/blitting.html)

        Parameters
        ----------
        None

        """
        plt.sca(self.ax)
        self.fig = self.ax.figure
        self.ax.set_aspect("equal")
        plt.show(block=False)
        plt.pause(0.1)
        self.fig_bg = self.fig.canvas.copy_from_bbox(self.fig.bbox)
        self.fig.canvas.blit(self.fig.bbox)


class SUMOScenario:
    def __init__(
        self,
        network_file,
        bicycle_type="Bicycle",
        animate=False,
        t_s=0.01,
        run_time_factor=1.0,
    ):
        """Create a Scenario object based on a SUMO network file (.net.xml)

        Parameters
        ----------
        network_file : str
            Path + filename to a SUMO network file
        bicycle_type : str
            Type of dynamic bicycle Model used within this scenario. Must be
            any of ('Bicycle', 'TwoDBicycle', 'InvPendulumBicycle'). Default is
            "Bicycle"
        animate : boolean,
            If True, runs a matplotlib animation of the simulated intersections
            parallel to SUMO.
        animate_save : boolean,
        t_s : float
            Simulation step lenght. Default is 0.01
        run_time_factor : float,
            Factor limiting the maximum simulation step lenght to
            run_time_factor * t_s with t_s beeing the simulated step lenght.
            Set to 'None' to run as fast as possible. Default is 1.0
        """

        # time utilities
        self.hist_run_time = []
        self.run_time_factor = run_time_factor
        self.t_s = t_s

        # parse bicyle type
        self.BICYCLE_TYPES = ("Bicycle", "TwoDBicycle", "InvPendulumBicycle")
        assert bicycle_type in self.BICYCLE_TYPES, (
            f"Parameter bicycle_type has to be any of "
            f"{self.BICYCLE_TYPES}, instead it was '{bicycle_type}'."
        )
        self.bicycle_type = bicycle_type

        # import network file
        self.intersections = []
        net = sumolib.net.readNet(network_file, withInternal=True)
        nodes = net.getNodes()

        # count nodes that are not dead ends
        n = 0
        for i in range(len(nodes)):
            if (
                len(nodes[i].getIncoming()) < 2
                and len(nodes[i].getOutgoing()) < 2
            ):
                continue
            n += +1

        # set up animation
        self.animate = animate
        if self.animate:
            nrows = floor(sqrt(n))
            ncols = ceil(sqrt(n))
            self.fig = plt.figure()
            j = 1

        # create intersections for SFM modelling
        for i in range(len(nodes)):
            # only include nodes that are not dead ends
            if (
                len(nodes[i].getIncoming()) < 2
                and len(nodes[i].getOutgoing()) < 2
            ):
                continue

            if self.animate:
                ax = self.fig.add_subplot(nrows, ncols, j)
                j += 1

                self.intersections.append(
                    SocialForceIntersection(
                        [],
                        animate=True,
                        axes=ax,
                        activate_sumo_cosimulation=True,
                        id=nodes[i].getID(),
                        net=net,
                    )
                )
                figManager = plt.get_current_fig_manager()
                figManager.resize(960, 1080)
                plt.show(block=False)
                plt.pause(0.1)
                self.fig_bg = self.fig.canvas.copy_from_bbox(self.fig.bbox)
                self.fig.canvas.blit(self.fig.bbox)
            else:
                self.intersections.append(
                    SocialForceIntersection(
                        [],
                        activate_sumo_cosimulation=True,
                        id=nodes[i].getID(),
                        net=net,
                    )
                )

    def allocate_road_users(self):
        """
        Allocate road users in the network to their intersections.

        Returns
        -------
        None.

        """
        for ins in self.intersections:
            ruids_entered, ruids_exited = ins.find_entered_exited_roadusers()

            # remove exited
            # ins.removeExited()
            ins.remove_road_users_by_id(ruids_exited)

            # add entered road users
            for i in ruids_entered:
                route = traci.vehicle.getRoute(i)
                current_route_index = traci.vehicle.getRouteIndex(i)
                route = route[current_route_index:]

                if len(route) < 2:
                    raise ValueError(
                        f"Road user {i} does not have"
                        "a valid remaining route with more then one "
                        "element. The invalid route was "
                        f"'{traci.vehicle.getRoute(i)}' with "
                        "current_route_index = {current_route_index}"
                    )

                pos = traci.vehicle.getPosition(i)
                s = [
                    pos[0],
                    pos[1],
                    angleSUMOtoSFM(traci.vehicle.getAngle(i)),
                    traci.vehicle.getSpeed(i),
                    0.0,
                ]

                if self.bicycle_type == self.BICYCLE_TYPES[0]:
                    params = BicycleParameters(t_s=self.t_s)
                    unew = Bicycle(s, i, route, params=params)
                elif self.bicycle_type == self.BICYCLE_TYPES[1]:
                    params = InvPendulumBicycleParameters(t_s=self.t_s)
                    unew = TwoDBicycle(s, i, route, params=params)
                elif self.bicycle_type == self.BICYCLE_TYPES[2]:
                    s.append(0.0)
                    params = InvPendulumBicycleParameters(t_s=self.t_s)
                    unew = InvPendulumBicycle(s, i, route, params=params)
                else:
                    raise ValueError(
                        f"Unknown bicycle type '{self.bicycle_type}'! Known"
                        f"types are {self.BICYCLE_TYPES}"
                    )
                ins.add_road_user(unew)

    def step(self, i):
        """Simulate a C-SFM step for all intersections in the scenario"""

        t = time()

        # Allocate road users on intersection to simulation by the C-SFM model
        # and road users on links to simulation by SUMO
        self.allocate_road_users()

        # Simulate intersections
        if self.animate:
            self.fig.canvas.restore_region(self.fig_bg)
        for ins in self.intersections:
            ins.step()
        if self.animate:
            self.fig.canvas.blit(self.fig.bbox)
            self.fig.canvas.flush_events()

        # SUMO step
        traci.simulationStep()

        # timing
        dt = time() - t
        if self.run_time_factor is not None:
            if dt < self.t_s * self.run_time_factor:
                sleep(self.t_s * self.run_time_factor - dt)
        self.hist_run_time.append(time() - t)

        if not self.animate:
            return i + 1

    def run(self, n_steps=None):
        """Run scenario simulation"""

        try:
            i = 0
            while traci.simulation.getMinExpectedNumber() > 0:
                if i == n_steps:
                    break
                self.step(i)
                i = i + 1
        except Exception:
            print(traceback.format_exc())
        finally:
            traci.close()
            sys.stdout.flush()

    def plot_runtime_vs_nvec(self, fig=None):
        """
        Plot a scatter plot of the simulation step duration vs. the number
        of vehicles per intersection.

        Parameters
        ----------
        fig : figure, optional
            Figure to be plotted in. The default is None and creates a new
            figure.

        Returns
        -------
        None.

        """
        if fig is None:
            plt.figure()
        n_max = 0
        t_max = np.max(self.hist_run_time)
        for ins in self.intersections:
            n_max = max(n_max, np.max(ins.hist_n_vecs))
            plt.scatter(
                self.hist_run_time, ins.hist_n_vecs, color="b", alpha=0.1
            )
        plt.plot(
            (self.t_s, self.t_s),
            (0, n_max),
            color="r",
            label="real-time requirement",
        )
        if self.run_time_factor is not None:
            plt.plot(
                (
                    self.run_time_factor * self.t_s,
                    self.run_time_factor * self.t_s,
                ),
                (0, n_max),
                color="r",
                linestyle="--",
                label="selected min. duration",
            )

        plt.title(
            (
                "Simulation step duration with "
                f"{'libsumo' if cfg.sumo_use_libsumo else 'traci'} and "
                f"{'with' if self.animate else 'without'} animation."
                f"Total time: {len(self.hist_run_time)*self.t_s} s, "
                f"{len(self.intersections)} intersection(s)."
            )
        )
        plt.xlabel("Duration of one simulation step [s]")
        plt.ylabel("Number of vehicles per intersection")

        plt.ylim(0 - 0.2, n_max + 0.2)
        plt.xlim(0, t_max + 0.002)

        plt.legend()
