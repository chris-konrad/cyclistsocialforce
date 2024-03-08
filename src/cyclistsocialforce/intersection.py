# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 16:48:35 2023.

@author: Christoph Schmidt
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.path as pth
from matplotlib.patches import PathPatch, Polygon
from scipy import interpolate

import cyclistsocialforce.config as cfg
from cyclistsocialforce.trajectory import generateSplinePrototype
from cyclistsocialforce.utils import (
    angleDifference,
    angleSFMtoSUMO,
    limitAngle,
    limitMagnitude,
)

# Imports that are only needed if the model ist used with SUMO
if cfg.has_sumo:
    if cfg.sumo_use_libsumo:
        import libsumo as traci
    else:
        import traci


class RoadSegment:
    def __init__(self, x0, beta0, width, ds=0.1):
        self.x0 = x0
        self.beta0 = beta0
        self.width = width
        self.edges = []
        self.ds = ds

    def calcRepulsiveForce(self, x, y):
        s = x.shape
        x = x.flatten()
        y = y.flatten()

        Fx = np.zeros_like(x)
        Fy = np.zeros_like(y)

        for e in self.edges:
            Fx_e, Fy_e = e.calcRepulsiveForce(x, y)
            Fx += Fx_e
            Fy += Fy_e

        return np.reshape(Fx, s), np.reshape(Fy, s)


class StraightRoadSegment(RoadSegment):
    def __init__(self, x0, beta0, width, length):
        RoadSegment.__init__(self, x0, beta0, width)
        self.length = length

        # edge coordinates
        x = np.arange(0, length, self.ds)
        yr = -(width / 2) * np.ones_like(x)
        yl = (width / 2) * np.ones_like(x)

        # rotation matrix
        R = np.array(
            [[np.cos(beta0), -np.sin(beta0)], [np.sin(beta0), np.cos(beta0)]]
        )
        
        # edge vcertices
        vert_r = R @ np.c_[x, yr].T + np.reshape(x0, (2, 1))
        vert_l = R @ np.c_[x, yl].T + np.reshape(x0, (2, 1))

        self.edges.append(RoadEdge(vert_r.T))
        self.edges.append(RoadEdge(vert_l.T))

        self.x1 = x0 + self.length * np.array([np.cos(beta0), np.sin(beta0)])

class CurvedRoadSegment(RoadSegment):
    def __init__(self, x0, beta0, width, radius, angle, direction):
        RoadSegment.__init__(self, x0, beta0, width)
        self.length = length

        #number of points
        npoints_r = 
        npoints_l = 
        
        # edge coordinates
        x = np.arange(0, length, self.ds)
        yr = -(width / 2) * np.ones_like(x)
        yl = (width / 2) * np.ones_like(x)

        # rotation matrix
        R = np.array(
            [[np.cos(beta0), -np.sin(beta0)], [np.sin(beta0), np.cos(beta0)]]
        )
        
        # edge vcertices
        vert_r = R @ np.c_[x, yr].T + np.reshape(x0, (2, 1))
        vert_l = R @ np.c_[x, yl].T + np.reshape(x0, (2, 1))

        self.edges.append(RoadEdge(vert_r.T))
        self.edges.append(RoadEdge(vert_l.T))

        self.x1 = x0 + self.length * np.array([np.cos(beta0), np.sin(beta0)])

class RoadEdge:
    """
    Road edges that exert a repulisive force on road users.

    WARNING: Under development! Highly experimental and not fit for purpose
    at the current time.
    """

    def __init__(self, vertices):
        self.vertices = vertices

    def calcRepulsiveForce(self, x, y):
        F0 = 0.01
        decay = -2.5

        s = x.shape
        x = x.flatten()
        y = y.flatten()

        r = np.sqrt(
            (self.vertices[:, 0][np.newaxis, :] - x[:, np.newaxis]) ** 2
            + (self.vertices[:, 1][np.newaxis, :] - y[:, np.newaxis]) ** 2
        )
        erx = (self.vertices[:, 0][np.newaxis, :] - x[:, np.newaxis]) / r
        ery = (self.vertices[:, 1][np.newaxis, :] - y[:, np.newaxis]) / r

        F = -F0 * r**decay
        Fx = np.sum(F * erx, axis=1)
        Fy = np.sum(F * ery, axis=1)

        return np.reshape(Fx, s), np.reshape(Fy, s)


class SocialForceIntersection:
    """
    Manages a social force intersection or open space. Works toghether with
    SUMO and as a standalone.
    """

    def __init__(
        self,
        vehicleList,
        id="",
        priority_rule="unregulated",
        animate=False,
        axes=None,
        activate_sumo_cosimulation=False,
        net=None,
        roadEdges=[],
    ):
        """
        Create a SocialForceIntersection.

        Use activate_sumo_cosimulation and provide a sumolib network object
        to run co-simulation with SUMO. Provides functions for road user
        allocation during simulation with SUMO. When deactivated,
        SocialForceIntersection manages road users on a boundless open space.

        Parameters
        ----------
        vehicleList : list(cyclistsocialforce.vehicle.Vehicle)
            List of vehicles on the intersection during creation. May be empty.
        id : str, optional
            ID/name of the intersection. The default is "".
        priority_rule : str, optional
            Priority rule of this intersection. Must be any of ("p2r",
            "unregulated"). The default is "unregulated".
        animate : boolean, optional
            If true, the intersection creates a matplotlib animation of the
            road user movements on it's surface. The default is False.
        axes : axes, optional
            Axes object for the animation. Required if animate=True.
            The default is None.
        activate_sumo_cosimulation : boolean, optional
            If true, the intersection is compatible to co-simulation with
            SUMO. The default is False.
        net : sumolib.net.Net, optional
            Sumolib network object holding information on the in-/out-edges and
            internal lanes of the intersection. The default is None.
        roadEdges : list(cyclistsocialforce.intersection.RoadEdge), optional
            List of road edges of this intersection. May be empty.
            The default is [].

        Returns
        -------
        None.

        """
        self.activate_sumo_cosimulation = activate_sumo_cosimulation
        self.vehicles = vehicleList
        self.n_bikes = len(vehicleList)

        self.vehicleX = np.zeros((self.n_bikes, 1))
        self.vehicleY = np.zeros((self.n_bikes, 1))
        self.vehicleTheta = np.zeros((self.n_bikes, 1))
        self.update_road_user_positions()

        self.animate = animate

        self.hist_n_vecs = []

        self.priority_rule = priority_rule

        assert isinstance(id, str), "Intersection ID has to be a string."
        self.id = id

        # SUMO
        if self.activate_sumo_cosimulation:
            self.node = net.getNode(id)

            # Intersection footprint
            self.shape = pth.Path(self.node.getShape(), closed=True)
            x, y = self.node.getCoord()

            # Incoming and outgoing edges
            self.inEdges = {}
            self.outEdges = {}

            for e in self.node.getIncoming():
                lanes = e.getLanes()
                self.inEdges[e.getID()] = []
                for l in lanes:
                    # identify the closest point to the node center
                    path = np.array(l.getShape())

                    tck, u = interpolate.splprep(
                        (path[:, 0], path[:, 1]),
                        s=0.0,
                        k=min(5, np.shape(path)[0] - 1),
                    )
                    x_i, y_i = interpolate.splev(np.linspace(0, 1, 10), tck)

                    x_i = x_i[-2:]
                    y_i = y_i[-2:]

                    self.inEdges[e.getID()].append((x_i, y_i))

            for e in self.node.getOutgoing():
                lanes = e.getLanes()
                self.outEdges[e.getID()] = []
                for l in lanes:
                    # identify the closest point to the node center
                    path = np.array(l.getShape())

                    tck, u = interpolate.splprep(
                        (path[:, 0], path[:, 1]),
                        s=0.0,
                        k=min(3, np.shape(path)[0] - 1),
                    )
                    x_i, y_i = interpolate.splev(np.linspace(0, 1, 10), tck)

                    x_i = x_i[:2]
                    y_i = y_i[:2]

                    self.outEdges[e.getID()].append((x_i, y_i))

            # create a list of internal lanes of this intersection.
            self.internal_lanes = []
            self.internal_lane_ids = []
            for e in net.getEdges():
                if e.getFromNode() == self.node and e.getToNode() == self.node:
                    lanes = e.getLanes()
                    for l in lanes:
                        self.internal_lane_ids.append(l.getID())
                        self.internal_lanes.append(l)

            if len(self.internal_lanes) == 0:
                raise ValueError(
                    (
                        f"Intersection {self.id} does not have "
                        "internal lanes! Cyclistsocialforce requires"
                        " internal lanes to allocate SUMO road users"
                        " to intersections. Check if the net-file "
                        "correctly specifies interal lanes for this "
                        "intersection!"
                    )
                )

        # List of additional road edges
        self.edges = roadEdges

        self.ax = axes

        if animate:
            assert self.ax is not None, "Provide axes for animation!"
            self.ghandles = []
            self.prepareAxes()
            for e in self.edges:
                patch = Polygon(e.vertices, closed=False, edgecolor="black")
                self.ax.add_patch(patch)
            if self.activate_sumo_cosimulation:
                patch = PathPatch(
                    self.shape, facecolor="black", edgecolor="black"
                )
                self.ax.add_patch(patch)

    def find_entered_exited_roadusers(self):
        """
        Find road users that entered/exited the intersection during last step.

        Requires SUMO/TRACI

        Returns
        -------
        ruids_enterd : list(str)
            IDs of road users that entered the intersection in the last step.
        ruids_exited : TYPE
            IDs of road users hat exited the intersection in the last step.

        """
        ruids_on_ins_prev = self.get_road_user_ids()
        ruids_on_ins_curr = []
        for l in self.internal_lane_ids:
            ruids_on_ins_curr += traci.lane.getLastStepVehicleIDs(l)

        ruids_exited = np.setdiff1d(ruids_on_ins_prev, ruids_on_ins_curr)
        ruids_entred = np.setdiff1d(ruids_on_ins_curr, ruids_on_ins_prev)

        # print(ruids_exited)

        return ruids_entred, ruids_exited

    def addEdge(self, roadEdge):
        self.edges.append(roadEdge)

    def add_road_user(self, user):
        """Add a single road user to the current intersection.

        If the road user has route following activated and the simulation
        operates with SUMO, this function also determines the trajectory
        protoype for the user.

        Parameters
        ----------
        user : Vehicle
            Vehicle object to be added to this intersection
        """

        # determin destinations according to vehicle route
        if self.activate_sumo_cosimulation & user.follow_route:
            ecurrent = user.route[0]
            enext = user.route[1]

            assert (
                ecurrent in self.inEdges
            ), f"Road user {user.id} arriving \
                on junction {self.id} from unknown edge {ecurrent}!"
            assert (
                enext in self.outEdges
            ), f"Road user {user.id} requesting \
                to depart junction {self.id} on unknown edge {enext}!"

            # determine closest incoming lane
            lanepoints = self.inEdges[ecurrent]
            if len(lanepoints) > 1:
                # more then 1 ingoing lane
                xpoints = np.concatenate((lanepoints[0][0], lanepoints[1][0]))
                ypoints = np.concatenate((lanepoints[0][1], lanepoints[1][1]))
                d = np.sqrt(
                    (xpoints - user.s[0]) ** 2 + (ypoints - user.s[1]) ** 2
                )
                laneid_in = int(np.argmin(d) / 2)
            else:
                laneid_in = 0

            # randomly select outgoing lane
            laneid_out = np.random.randint(0, len(self.outEdges[enext]))

            # generate spline prototype between selected lanes
            points = np.vstack(
                (
                    np.array(self.inEdges[ecurrent][laneid_in]).T,
                    np.array(self.outEdges[enext][laneid_out]).T,
                )
            )

            xp, yp = generateSplinePrototype(points[:, 0], points[:, 1], 5)

            # remove points behind the road user
            dp2f = np.sqrt((xp - xp[-1]) ** 2 + (yp - yp[-1]) ** 2)
            du2f = np.sqrt(
                (user.s[0] - xp[-1]) ** 2 + (user.s[1] - yp[-1]) ** 2
            )
            xp = xp[dp2f < du2f]
            yp = yp[dp2f < du2f]

            # set destinations
            user.setDestinations(xp, yp, reset=True)

        # add road user to intersection
        print(f"Adding {user.id} to intersection {self.id}")
        self.vehicles.append(user)
        if self.n_bikes > 0:
            self.vehicleX = np.vstack((self.vehicleX, (user.s[0],)))
            self.vehicleY = np.vstack((self.vehicleY, (user.s[1],)))
            self.vehicleTheta = np.vstack((self.vehicleTheta, (user.s[2],)))
        else:
            self.vehicleX = np.array(((user.s[0],),))
            self.vehicleY = np.array(((user.s[1],),))
            self.vehicleTheta = np.array(((user.s[2],),))
        self.n_bikes += 1

    def get_road_user_ids(self):
        """
        Return a list if road user ids currently on the intersection.

        Returns
        -------
        ru_ids : list(str)
            Road user ids.

        """

        return [v.id for v in self.vehicles]

    def has_road_user(self, userId):
        """Check if a user with given ID is simulated by this intersection.

        Finds the first road user if multiple RUs with the same name are
        present.

        Parameters
        ----------
        userId : str
            User ID given by TraCI.

        Returns
        -------
        has_road_user : boolean
            True if the intersection has a road user with id userId.
        """
        assert isinstance(userId, str), "User ID has to be a string."

        ids = self.get_road_user_ids()

        return userId in ids

    def remove_road_users_by_id(self, ruids):
        """
        Remove the road users given by a list of ids from the intersection.

        Parameters
        ----------
        ruids : list(str)
            List of road user ids to be removed.

        Returns
        -------
        None.

        """

        if len(ruids):
            i_remove = np.zeros((len(ruids)), dtype=int)

            # remove road users from vehicle lists
            vehicles_remaining = []
            i = 0
            j = 0
            for v in self.vehicles:
                if v.id not in ruids:
                    vehicles_remaining.append(v)
                else:
                    i_remove[j] = i
                    j += 1
                    print(f"Removing {v.id} from intersection {self.id}")
                i += 1
            if j > 0:
                self.vehicles = vehicles_remaining
                self.n_bikes = self.n_bikes - len(ruids)
                self.vehicleX = np.delete(self.vehicleX, i_remove, 0)
                self.vehicleY = np.delete(self.vehicleY, i_remove, 0)
                self.vehicleTheta = np.delete(self.vehicleTheta, i_remove, 0)

        # remove animation handles
        if self.n_bikes == 0 and self.animate:
            while len(self.ghandles) > 0:
                self.ghandles.pop(0).remove()

    def remove_road_user(self, i_remove):
        """Manually remove the road user i from the list of vehicles."""

        self.vehicles = [
            self.vehicles[i]
            for i in range(len(self.vehicles))
            if not i == i_remove
        ]
        self.n_bikes = self.n_bikes - 1

        self.vehicleX = np.delete(self.vehicleX, i_remove, 0)
        self.vehicleY = np.delete(self.vehicleY, i_remove, 0)
        self.vehicleTheta = np.delete(self.vehicleTheta, i_remove, 0)

        if self.n_bikes == 0 and self.animate:
            while len(self.ghandles) > 0:
                self.ghandles.pop(0).remove()

    def prepareAxes(self):
        plt.sca(self.ax)
        if self.activate_sumo_cosimulation:
            plt.xlim(
                [
                    np.amin(self.shape.vertices[:, 0]) - 1,
                    np.amax(self.shape.vertices[:, 0]) + 1,
                ]
            )
            plt.ylim(
                [
                    np.amin(self.shape.vertices[:, 1]) - 1,
                    np.amax(self.shape.vertices[:, 1]) + 1,
                ]
            )
        else:
            pass
            # plt.xlim([0, 12])
            # plt.ylim([0, 12])
        # plt.title('Social Force Simulation')
        # plt.xlabel(r'$\frac{\it{x}}{\mathrm{m}}$')
        # plt.ylabel(r'$\frac{\it{y}}{\mathrm{m}}$')
        self.ax.set_aspect("equal", adjustable="box")

    def update_road_user_positions(self):
        """
        Update road user positions.

        Updates the intersections list of road user positions. When co-
        simulating with SUMO, send road user positions to SUMO via
        traci.vehicle.moveToXY()

        Returns
        -------
        None.

        """

        for i in range(0, self.n_bikes):
            self.vehicleX[i, 0] = self.vehicles[i].s[0]
            self.vehicleY[i, 0] = self.vehicles[i].s[1]
            self.vehicleTheta[i, 0] = self.vehicles[i].s[2]

            if self.activate_sumo_cosimulation:
                traci.vehicle.moveToXY(
                    self.vehicles[i].id,
                    "",
                    -1,
                    self.vehicles[i].s[0],
                    self.vehicles[i].s[1],
                    angle=angleSFMtoSUMO(self.vehicles[i].s[2]),
                    keepRoute=6,
                )

    def get_untracked_foes(self):
        """
        Determine which road users are not considered by the ego road user.

        Deternines untracked foes based on the ego FOV and simple priority-
        to-the-right based on relative azimuth.

        Returns
        -------
        foes_untracked : array(boolean)
            Boolean array indicating which road users are not tracked. First
            dimension determines the ego road user. Boolean True in the
            second dimension marks that this road user is not tracked by
            the ego road user.

        """
        if self.n_bikes > 1:
            # Identify foes of every road user.
            foes_untracked = np.full((self.n_bikes, self.n_bikes), False)

            # Relative Angles of foe vehicles
            dX = np.tile(self.vehicleX.T, self.n_bikes) - np.tile(
                self.vehicleX, (self.n_bikes, 1)
            )
            dY = np.tile(self.vehicleY.T, self.n_bikes) - np.tile(
                self.vehicleY, (self.n_bikes, 1)
            )

            foe_azimuth_abs = limitAngle(np.arctan2(dY, dX).T)
            foe_azimuth_rel = np.zeros_like(foe_azimuth_abs)
            ego_yaw = np.tile(self.vehicleTheta, self.n_bikes).T

            for i in range(self.n_bikes):
                for j in range(self.n_bikes):
                    foe_azimuth_rel[i, j] = angleDifference(
                        ego_yaw[i, j], foe_azimuth_abs[i, j]
                    )

                    # Can't be your own foe.
                    if i == j:
                        foes_untracked[i, j] = True

                    # Foe outside field of view.
                    if abs(foe_azimuth_rel[i, j]) > (
                        self.vehicles[i].params.hfov / 2
                    ):
                        foes_untracked[i, j] = True

                    # Priority to the right based on relative azimuth.
                    if self.priority_rule == "p2r":
                        if foe_azimuth_rel[i, j] > 0:
                            foes_untracked[i, j] = True
        else:
            foes_untracked = np.array((True))

        return foes_untracked

    def calc_forces(self):
        """
        Calculate the force matrices experienced by RUs on the intersection.

        Calculates and sums destination force with repulsive forces of other
        road users and road edges. Identifies tracked foes based on field-
        of-view and simple relative azimuth-based priority-to-the-right.

        Returns
        -------
        Fx : array(float)
            Magnitude of the total force in X direction.
        Fy : TYPE
            Magnitude of the total force in Y direction.

        """
        # maintain the right number of arrow handles for drawing
        if self.animate:
            while len(self.ghandles) < self.n_bikes**2:
                self.ghandles.append(
                    self.ax.arrow(
                        0,
                        0,
                        1,
                        1,
                        head_width=0.3,
                        head_length=0.4,
                        linewidth=1,
                        edgecolor="gray",
                        facecolor="gray",
                        animated=True,
                        zorder=3,
                    )
                )

            while len(self.ghandles) > self.n_bikes**2:
                self.ghandles.pop(0).remove()

            a = 0

        # identify untracked foes
        foes_untracked = self.get_untracked_foes()

        # init force matrices
        Fx = np.zeros((self.n_bikes, self.n_bikes))
        Fy = np.zeros((self.n_bikes, self.n_bikes))
        Fdestx = np.zeros((self.n_bikes))
        Fdesty = np.zeros((self.n_bikes))

        # calc forces for every vehicle
        for i in range(0, self.n_bikes):
            # calc destination force
            Fdestx[i], Fdesty[i] = self.vehicles[i].calcDestinationForce()

            # draw destination force
            if self.animate:
                self.ghandles[a].set_data(
                    x=self.vehicleX[i, 0],
                    y=self.vehicleY[i, 0],
                    dx=Fdestx[i] / 2,
                    dy=Fdesty[i] / 2,
                )

                a += 1

            # calc repulsive forces
            if self.n_bikes > 1:
                Fxi, Fyi = self.vehicles[i].calcRepulsiveForce(
                    np.delete(self.vehicleX, np.where(foes_untracked[i, :])),
                    np.delete(self.vehicleY, np.where(foes_untracked[i, :])),
                    np.delete(
                        self.vehicleTheta, np.where(foes_untracked[i, :])
                    ),
                )

                Fx[i, np.logical_not(foes_untracked[i, :])] = Fxi
                Fy[i, np.logical_not(foes_untracked[i, :])] = Fyi

        if self.n_bikes > 1:
            # draw repulsive forces
            for i in range(self.n_bikes):
                for j in range(self.n_bikes):
                    if self.animate and i != j:
                        self.ghandles[a].set_data(
                            x=self.vehicleX[j, 0],
                            y=self.vehicleY[j, 0],
                            dx=Fx[i, j],
                            dy=Fy[i, j],
                        )
                        self.ax.draw_artist(self.ghandles[a])

                        a += 1

            # Sum repulsive forces and destination force
            Frepx, Frepy = limitMagnitude(
                np.sum(Fx, axis=0),
                np.sum(Fy, axis=0),
                np.sqrt(Fdestx**2 + Fdesty**2),
            )

            Fx = Frepx + Fdestx
            Fy = Frepy + Fdesty
        else:
            Fx = Fdestx
            Fy = Fdesty

        # Add edge forces
        for e in self.edges:
            Fxe, Fye = e.calcRepulsiveForce(self.vehicleX, self.vehicleY)
            Fx += Fxe.flatten()
            Fy += Fye.flatten()

        # Set forces to vehicles
        for i in range(0, self.n_bikes):
            self.vehicles[i].force = (Fx[i], Fy[i])
            self.vehicles[i].F.append(np.sqrt(Fx[i] ** 2 + Fy[i] ** 2))

        return Fx, Fy

    def step(self):
        """
        Execute one simulation step for this intersection.

        This calculates all social forces and calls each road user on the
        intersection to execute one control loop step based on these forces.
        Finally, update the road user locations.

        Returns
        -------
        None.

        """
        if self.n_bikes > 0:
            Fx, Fy = self.calc_forces()

            for i in range(0, self.n_bikes):
                self.vehicles[i].step(Fx[i], Fy[i])
                if self.animate:
                    if self.vehicles[i].hasDrawings[0]:
                        self.vehicles[i].updateBikeDrawing(self.ax)
                    else:
                        self.vehicles[i].makeBikeDrawing(
                            self.ax,
                            drawTrajectory=True,
                            drawNextDest=True,
                            drawDestQueue=True,
                            drawPastDest=True,
                            drawForce=True,
                            drawName=True,
                        )

            self.update_road_user_positions()

        self.hist_n_vecs.append(self.n_bikes)

    def endAnimation(self):
        """End animation of the intersection.

        Set the "animated" property of all graphic object to False to prevent
        them from disappearing once the animation ends.

        Returns
        -------
        None.

        """
        if self.animate:
            for g in self.ghandles:
                g.set_animated(False)
            for v in self.vehicles:
                v.endAnimation()
            self.animate = False

    def restartAnimation(self):
        """End animation of the intersection.

        Set the "animated" property of all graphic object to True to enable
        animation.

        Returns
        -------
        None.

        """
        if not self.animate:
            for g in self.ghandles:
                g.set_animated(True)
            for v in self.vehicles:
                v.restartAnimation()
            self.animate = True
