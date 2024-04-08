# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 13:08:17 2023.

Classes and functions for the vizualisation of the simulation.

@author: Christoph M. Schmidt
"""

import numpy as np

from matplotlib.patches import Polygon
from matplotlib.collections import PolyCollection
from matplotlib.lines import Line2D
from pypaperutils.design import TUDcolors

from mpl_toolkits.mplot3d.art3d import Line3D, Poly3DCollection

from cyclistsocialforce.parameters import BikeDrawing2DParameters


class VehicleDrawing:
    def __init__(
        self,
        ax,
        vehicle,
        draw_force_resulting=True,
        draw_force_destination=True,
        draw_forces_repulsive=True,
        draw_trajectory=False,
        draw_nextdest=False,
        draw_destqueue=False,
        draw_pastdest=False,
        draw_name=False,
        animated=True,
    ):
        """Drawing of the peripherals common to all road users"""

        self.animated = animated
        self.ax = ax

        self.draw_force_resulting = draw_force_resulting
        self.draw_force_destination = draw_force_destination
        self.draw_forces_repulsive = draw_forces_repulsive
        self.draw_trajectory = draw_trajectory
        self.draw_nextdest = draw_nextdest
        self.draw_pastdest = draw_pastdest
        self.draw_destqueue = draw_destqueue
        self.draw_name = draw_name

        self.ghandles = {}

        self.make_drawing()

    def make_drawing(self, vehicle):
        """Create the graphics handles for the elements of the drawing.

        Parameters
        ----------
        vehicle : cyclistsocialforce.vehicle
            Any vehicle from the vehicle module
        """

        self.make_force_arrows()

        if self.draw_trajectory:
            self.make_trajectory_drawing(vehicle)
        if self.draw_nextdest:
            self.make_nextdest_drawing(vehicle)
        if self.draw_pastdest:
            self.make_pastdest_drawing(vehicle)
        if self.draw_destqueue:
            self.make_destqueue_drawing(vehicle)
        if self.draw_name:
            self.make_name_drawing(vehicle)

    def make_trajectory_drawing(self, vehicle):
        """Creates a plot of the past x-y trajectory

        Parameters
        ----------
        vehicle : cyclistsocialforce.vehicle
            Any vehicle from the vehicle module
        """

        self.ghandles["trajectory"] = self.ax.plot(
            vehicle.traj[0],
            vehicle.traj[1],
            color=(0.0 / 255, 166.0 / 255, 214.0 / 255),
            linewidth=1,
            animated=self.animated,
        )
        self.ax.draw_artist(self.ghandles["trajectory"])

    def make_nextdest_drawing(self, vehicle):
        """Draw a straigt line to the next destination. Marks the next
        destination with an x.

        Parameters
        ----------
        vehicle : cyclistsocialforce.vehicle
            Any vehicle from the vehicle module
        """
        (self.ghandles["nextdest-line"],) = self.ax.plot(
            (vehicle.s[0], vehicle.dest[0]),
            (vehicle.s[1], vehicle.dest[1]),
            color="gray",
            linewidth=1,
            linestyle="dashed",
            animated=self.animated,
            zorder=3,
        )
        self.ax.draw_artist(self.ghandles["nextdest-line"])

        if not self.draw_destqueue:
            (self.ghandles["nextdest-marker"],) = self.ax.plot(
                vehicle.dest[0],
                vehicle.dest[1],
                marker="x",
                markersize=5,
                markeredgecolor="gray",
                markeredgewidth=2,
                animated=self.animated,
                zorder=3,
            )
            self.ax.draw_artist(self.ghandles["nextdest-marker"])

    def make_destqueue_drawing(self, vehicle):
        """Draw markers for the remaining intermediate destinations in the
        queue.

        Parameters
        ----------
        vehicle : cyclistsocialforce.vehicle
            Any vehicle from the vehicle module
        """

        if vehicle.destqueue is None:
            (self.ghandles["destqueue"],) = self.ax.plot(
                vehicle.dest[0],
                vehicle.dest[1],
                marker="x",
                markersize=5,
                markeredgecolor=(0.0 / 255, 166.0 / 255, 214.0 / 255),
                markeredgewidth=1,
                animated=self.animated,
                zorder=3,
            )
            self.ax.draw_artist(self.ghandles["destqueue"])
        else:
            (self.ghandles["destqueue"],) = self.ax.plot(
                vehicle.destqueue[vehicle.destpointer :, 0],
                vehicle.destqueue[vehicle.destpointer :, 1],
                linestyle="None",
                marker="x",
                markersize=5,
                markeredgecolor="gray",
                markeredgewidth=1,
                animated=self.animated,
                zorder=3,
            )
            self.ax.draw_artist(self.ghandles["destqueue"])

    def make_pastdest_drawing(self, vehicle):
        """Draw markers for the past destinations.


        Parameters
        ----------
        vehicle : cyclistsocialforce.vehicle
            Any vehicle from the vehicle module
        """
        if self.draw_pastdest and vehicle.destqueue is not None:
            (self.ghandles["pastdest"],) = self.ax.plot(
                vehicle.destqueue[: vehicle.destpointer, 0],
                vehicle.destqueue[: vehicle.destpointer, 1],
                linestyle="None",
                marker="x",
                markersize=5,
                markeredgecolor="gray",
                markeredgewidth=1,
                animated=self.animated,
                zorder=3,
            )
            self.ax.draw_artist(self.ghandles["pastdest"])

    def make_name_drawing(self, vehicle):
        """Draw the name of the vehicle.

        Parameters
        ----------
        vehicle : cyclistsocialforce.vehicle
            Any vehicle from the vehicle module
        """
        self.ghandles["name"] = self.ax.text(
            vehicle.s[0],
            vehicle.s[1] + 1,
            vehicle.id,
            color="black",
            fontsize=8,
            animated=self.animated,
            zorder=4,
        )
        self.ax.draw_artist(self.ghandles["name"])

    def make_force_arrows(self):
        """Create a set of lists to storing the handles of force arrows

        Parameters
        ----------
        None.

        Returns
        -------
        None.

        """

        if self.params.show_forces_destination:
            self.force_handle_dest = self.ax.arrow(
                0,
                0,
                0,
                0,
                head_width=0.3,
                head_length=0.4,
                linewidth=1,
                edgecolor=self.params.force_color_dest,
                facecolor=self.params.force_color_dest,
                animated=self.animated,
                zorder=3,
            )

        if self.params.show_forces_resulting:
            self.force_handle_res = self.ax.arrow(
                0,
                0,
                0,
                0,
                head_width=0.3,
                head_length=0.4,
                linewidth=1,
                edgecolor=self.params.force_color_res,
                facecolor=self.params.force_color_res,
                animated=self.animated,
                zorder=3,
            )

        if self.params.show_forces_repulsive:
            self.force_handles_rep = []

    def update(self, vehicle, Fdest=None, Frep=None, Fres=None):
        """Updates all elements of the vehicle drawing.

        Parameters
        ----------
        vehicle : cyclistsocialforce.vehicle
            Any vehicle from the vehicle module
        """

        self.update_forces(vehicle.s, Fdest, Frep, Fres)

        self.update_trajectory_drawing(vehicle)
        self.update_nextdest_drawing(vehicle)
        self.update_destqueue_drawing(vehicle)
        self.update_pastdest_drawing(vehicle)
        self.update_name_drawing(vehicle)

    def update_trajectory_drawing(self, vehicle):
        """Update the trajctory drawing

        Parameters
        ----------
        vehicle : cyclistsocialforce.vehicle
            Any vehicle from the vehicle module
        """
        if self.draw_trajectory:
            self.ghandles["trajectory"].set_data(
                vehicle.traj[0, 0 : vehicle.i], vehicle.traj[1, 0 : vehicle.i]
            )
            self.ax.draw_artist(self.ghandles["trajectory"])

    def update_nextdest_drawing(self, vehicle):
        """Update the drawing of the next destinations.

        Parameters
        ----------
        vehicle : cyclistsocialforce.vehicle
            Any vehicle from the vehicle module
        """
        if self.draw_nextdest:
            if vehicle.destspline is not None:
                self.ghandles["nextdest-line"].set_data(
                    (vehicle.destspline[:, 0]),
                    (vehicle.destspline[:, 1]),
                )
            else:
                self.ghandles["nextdest-line"].set_data(
                    (vehicle.s[0], vehicle.dest[0]),
                    (vehicle.s[1], vehicle.dest[1]),
                )
            self.ax.draw_artist(self.ghandles["nextdest-line"])

            if not self.draw_destqueue:
                self.ghandles["nextdest-marker"].set_data(
                    vehicle.dest[0], vehicle.dest[1]
                )
                self.ax.draw_artist(self.ghandles["nextdest-marker"])

    def update_destqueue_drawing(self, vehicle):
        """Update the drawing of the destination queue.

        Parameters
        ----------
        vehicle : cyclistsocialforce.vehicle
            Any vehicle from the vehicle module
        """
        if self.draw_destqueue:
            if vehicle.destqueue is None:
                self.ghandles["destqueue"].set_data(
                    vehicle.dest[0], vehicle.dest[1]
                )
            else:
                self.ghandles["destqueue"].set_data(
                    vehicle.destqueue[vehicle.destpointer :, 0],
                    vehicle.destqueue[vehicle.destpointer :, 1],
                )
            self.ax.draw_artist(self.ghandles["destqueue"])

    def update_pastdest_drawing(self, vehicle):
        """Update the drawing of the past destinations

        Parameters
        ----------
        vehicle : cyclistsocialforce.vehicle
            Any vehicle from the vehicle module
        """
        if self.draw_pastdest:
            if self.ghandles["pastdest"] is not None:
                self.ghandles["pastdest"].set_data(
                    vehicle.destqueue[: vehicle.destpointer, 0],
                    vehicle.destqueue[: vehicle.destpointer, 1],
                )
            self.ax.draw_artist(self.ghandles["pastdest"])

    def update_name_drawing(self, vehicle):
        """Update the drawing of the vehicle name

        Parameters
        ----------
        vehicle : cyclistsocialforce.vehicle
            Any vehicle from the vehicle module
        """
        if self.draw_name:
            self.ghandles[14].set_position((vehicle.s[0], vehicle.s[1] + 1))
            self.ax.draw_artist(self.ghandles[14])

    def update_forces(self, s, Fdest=None, Frep=None, Fres=None):
        """Updates the force vector drawing according to the currently
        experienced forces.

        Parameters
        ----------
        s : array-like, optional
            First two state of the bicycle (x,y). Not drawn if not specified.
        Fdest : array-like
            Destination force given as (Fx, Fy). Not updated if not specified.
        Frep : array-like
            List of repulisve forces given as ((Fx1, Fx2, ...) (Fy1, Fy2 ...)).
            Not updated if not specified.
        Fres : array-like
            Resulting force given as (Fx, Fy). Not updated if not specified.

        Returns
        -------
        None.

        """

        # repulsive forces
        if self.params.show_forces_repulsive and Frep is not None:
            n = len(Frep[0])

            while len(self.force_handles_rep) < n:
                self.force_handles_rep.append(
                    self.ax.arrow(
                        0,
                        0,
                        0,
                        0,
                        head_width=0.3,
                        head_length=0.4,
                        linewidth=1,
                        edgecolor=self.params.force_color_res,
                        facecolor=self.params.force_color_res,
                        animated=self.animated,
                        zorder=3,
                    )
                )
            if len(len(self.force_handles_rep)) > n:
                self.force_handle_rep = self.force_handle_rep[:n]

            for a, fx, fy in zip(self.force_handles_rep, Frep[0], Frep[1]):
                a.set_data(x=s[0], y=s[1], dx=fx, dy=fy)
                self.ax.draw_artist(a)

        # destination force
        if self.params.show_forces_destination and Fdest is not None:
            self.force_handle_dest.set_data(
                x=s[0], y=s[1], dx=Fdest[0], dy=Fdest[1]
            )
            self.ax.draw_artist(self.force_handle_dest)

        # resulting force
        if self.params.show_forces_resulting and Fres is not None:
            self.force_handle_res.set_data(
                x=s[0], y=s[1], dx=Fres[0], dy=Fres[1]
            )
            self.ax.draw_artist(self.force_handle_res)


class CarDrawing2D:
    def __init__(self, ax, car, animated=False):
        """Create a 2D Bicycle Drawing made of polygons.

        Parameters
        ----------
        ax : Axes
            Axes where the drawing should be created in.
        car : cyclistsocialforce.vehicle.StationaryIMPTCCar
            Car object to be drawn.
        animated : bool, optional
            Animate the drawing. The default is False.
        """

        self.has_fixed_dimensions = not isinstance(
            car, cyclistsocialforce.vehicle.StationaryIMPTCCar
        )
        self.animated = animated
        self.ax = ax

        self.make_polygon(car.s)

    def make_polygon(self, s):
        """Create the polygon collections that make the car drawing.

        Called by the constructor.

        Parameters
        ----------
        s : array-like
            Current car state as returned by car.s.

        Returns
        -------
        None.

        """
        keypoints = self.calc_keypoints(s)

        self.p = PolyCollection(
            keypoints,
            animated=self.animated,
            facecolors="gray",
            edgecolors="black",
            zorder=10,
        )
        self.ax.add_collection(self.p)

    def calc_keypoints(self, s):
        """Calculate the corners of the polygons.

        Uses fixed witdth, length and the current orientation if
        self.has_fixed_dimesion. Otherwise uses floating cornerns that
        implicitely determine the orientation and that are expected to be
        the last 8 entries in s (as returned by StationaryIMPTCCar.s)

        Parameters
        ----------
        s : array
            Car state.

        Returns
        -------
        keypoints : List[Array]
            List of arrays describing the corners of each polygon.
        """
        if self.has_fixed_dimensions:
            R_psi = np.array(
                [[np.cos(s[2]), -np.sin(s[2])], [np.sin(s[2]), np.cos(s[2])]]
            )
            keypoints = self.prototype_keypoints()
            keypoints = R_psi @ self.prototype_keypoints().T + np.array(
                [[s[0]], [s[1]]]
            )
        else:
            keypoints = [np.reshape(s[-8:], (4, 2))]
            keypoints.append(keypoints[[0, 2], :])
            keypoints.append(keypoints[[1, 3], :])

        return keypoints

    def update(self, car):
        """Update the drawing according to the car state.

        Parameters
        ----------
        bike : cyclistsocialforce.vehicle.StationaryIMPTCCar
            Car object whose state the drawing will be updated to.

        Returns
        -------
        None.

        """

        keypoints = self.calc_keypoints(car.s)

        self.p.set_verts(keypoints)
        self.ax.draw_artist(self.p)


class BicycleDrawing2D:
    """A 2D drawing of a standard bicyle with rider from bird-eyes view.

    TODO: Dimensions and Colors should be specifieable in the BicycleParameters
    object.
    """

    def __init__(
        self,
        ax,
        bike,
        params=None,
        proj_3d=False,
        animated=False,
        show_roll_indicator=True,
        show_force_resulting=True,
        show_force_destination=True,
        show_forces_repulsive=True,
    ):
        """Create a 2D Bicycle Drawing made of polygons.

        Parameters
        ----------
        ax : Axes
            Axes where the drawing should be created in.
        bike : cyclistsocialforce.vehicle.Bicycle
            Bicycle object to be drawn.
        params : cyclistsocialforce.parameters.BikeDrawing2DParameters, optinl
            Parameters object. If none is given, initializes the default
            parameters.
        proj_3d : bool. optinal
            Project the 2D drawing in the ground plane of a 3D plot. The
            default is False.
        animated : bool, optional
            Animate the drawing. The default is False.
        show_roll_indicator : bool, optional
            Draws a small roll angle indicator (2D: bubble scale, 3D: inv.
            pendulum). This only has an effect if the bicycle is a
            InvPendulumBicycle. The default is True.

        Returns
        -------
        None.

        """

        if show_roll_indicator is True:
            show_roll_indicator = type(bike).__name__ == "InvPendulumBicycle"

        if params is None:
            self.params = BikeDrawing2DParameters(
                show_roll_indicator=show_roll_indicator,
                proj_3d=proj_3d,
                show_force_resulting=show_force_resulting,
                show_force_destination=show_force_destination,
                show_forces_repulsive=show_forces_repulsive,
            )
        else:
            self.params = params
            if not show_roll_indicator:
                self.params.show_roll_indicator = False
                self.params.make_colorlists_riderbike()
            if proj_3d:
                self.params.proj_3d = True
                self.params.make_colorlists_riderbike()
            if not self.params.get_show_forces():
                self.params.show_forces_destination = False
                self.params.show_forces_repulsive = False
                self.params.show_forces_resulting = False

        self.l_1 = bike.params.l_1
        self.l_2 = bike.params.l_2
        self.ax = ax
        self.animated = animated

        self.make_polygon(bike.s)

    def make_polygon(self, s):
        """Create the polygon collections that make the bike drawing.

        Called by the constructor.

        Parameters
        ----------
        s : array-like
            Current bicycle state as returned by bicycle.s.

        Returns
        -------
        None.

        """
        keypoints = self.calc_keypoints(s)

        if self.params.proj_3d:
            self.p = Poly3DCollection(
                keypoints,
                facecolors=self.params.fcolors_riderbike,
                edgecolors=self.params.ecolors_riderbike,
                animated=False,
            )
            self.ax.add_collection3d(self.p, zs=0)
        else:
            self.p = PolyCollection(
                keypoints,
                animated=self.animated,
                facecolors=self.params.fcolors_riderbike,
                edgecolors=self.params.ecolors_riderbike,
                zorder=10,
            )
            self.ax.add_collection(self.p)

    def update(self, bike):
        """Update the drawing according to the bicycles state.

        Parameters
        ----------
        bike : cyclistsocialforce.vehicle.Bicycle
            Bicycle object whose state the drawing will be updated to.

        Returns
        -------
        None.

        """

        keypoints = self.calc_keypoints(bike.s)

        self.p.set_verts(keypoints)
        if self.params.show_roll_indicator:
            if abs(bike.s[5]) > np.pi / 4:
                self.params.ecolors_riderbike[-2] = self.params.tud_colors.get(
                    "rood"
                )
            else:
                self.params.ecolors_riderbike[-2] = "black"
            self.p.set(
                facecolor=self.params.fcolors_riderbike,
                edgecolor=self.params.ecolors_riderbike,
            )

        if self.params.proj_3d:
            self.p.do_3d_projection()
        else:
            self.ax.draw_artist(self.p)

    def calc_keypoints(self, s, w=0.45):
        """Calculate the corners of the polygons.

        TODO: Make dimensions class properties.

        Parameters
        ----------
        s : array
            Bicycle state.
        w : float, optional
            Handlebar width. The default is 0.45.

        Returns
        -------
        keypoints : List[Array]
            List of arrays describing the corners of each polygon.

        """
        R_psi = np.array(
            [[np.cos(s[2]), -np.sin(s[2])], [np.sin(s[2]), np.cos(s[2])]]
        )
        R_delta = np.array(
            [[np.cos(s[4]), -np.sin(s[4])], [np.sin(s[4]), np.cos(s[4])]]
        )
        R_delta2 = np.array(
            [
                [np.cos(s[4] / 2), -np.sin(s[4] / 2)],
                [np.sin(s[4] / 2), np.cos(s[4] / 2)],
            ]
        )

        rwhl = np.array(
            [
                [-self.l_1 - 0.65 / 2, 0.03],
                [-self.l_1 + 0.65 / 2, 0.03],
                [-self.l_1 + 0.65 / 2, -0.03],
                [-self.l_1 - 0.65 / 2, -0.03],
            ]
        )

        fwhl = R_delta @ np.array(
            [
                [-0.65 / 2, 0.03],
                [0.65 / 2, 0.03],
                [0.65 / 2, -0.03],
                [-0.65 / 2, -0.03],
            ]
        ).T + np.array([[self.l_2], [0]])

        hbar = R_delta @ np.array(
            [
                [-0.14 / 2, w / 2],
                [-0.06 / 2, w / 2],
                [-0.06 / 2, -w / 2],
                [-0.14 / 2, -w / 2],
            ]
        ).T + np.array([[self.l_2], [0]])

        hbar_temp = R_delta @ np.array(
            [
                [-0.14 / 2, w / 2 - 0.07],
                [-0.06 / 2, w / 2 - 0.07],
                [-0.06 / 2, -w / 2 + 0.07],
                [-0.14 / 2, -w / 2 + 0.07],
            ]
        ).T + np.array([[self.l_2], [0]])

        frme = np.array(
            [
                [-self.l_1, 0.02],
                [self.l_2, 0.02],
                [self.l_2, -0.02],
                [-self.l_1, -0.02],
            ]
        )

        body = np.array(
            [
                [-0.2 * np.sin(s[4] / 2) + 0.1, 0.2 * np.cos(s[4] / 2)],
                [0.2 * np.sin(s[4] / 2) + 0.1, -0.2 * np.cos(s[4] / 2)],
                [-0.75 * self.l_1, -0.15],
                [-0.75 * self.l_1, 0.15],
            ]
        )

        rarm = np.array(
            [
                [-0.2 * np.sin(s[4] / 2), 0.2 * np.cos(s[4] / 2)],
                hbar[:, 1],
                hbar_temp[:, 1],
                [-0.1 * np.sin(s[4] / 2), 0.1 * np.cos(s[4] / 2)],
            ]
        )

        larm = np.array(
            [
                [0.2 * np.sin(s[4] / 2), -0.2 * np.cos(s[4] / 2)],
                hbar[:, 2],
                hbar_temp[:, 2],
                [0.1 * np.sin(s[4] / 2), -0.1 * np.cos(s[4] / 2)],
            ]
        )

        head = (
            R_delta2
            @ np.array([[0.1, 0.1], [0.1, -0.1], [-0.1, -0.1], [-0.1, 0.1]]).T
        )

        keypoints = [rwhl, fwhl.T, frme, hbar.T, body, rarm, larm, head.T]

        for i in range(len(keypoints)):
            keypoints[i] = R_psi @ keypoints[i].T + np.array([[s[0]], [s[1]]])
            keypoints[i] = keypoints[i].T

            # if 3d, add zeros for z-axis.
            if self.params.proj_3d:
                keypoints[i] = np.concatenate(
                    (keypoints[i], np.array([[0], [0], [0], [0]])), axis=1
                )

        # if 3d, additionally create a stylized 3d penulum to the 2d drawing.
        if self.params.show_roll_indicator:
            if self.params.proj_3d:
                pend = R_psi @ np.array(
                    [
                        [-0.1, 0],
                        [-0.1, np.sin(s[5])],
                        [0.1, np.sin(s[5])],
                        [0.1, 0],
                    ]
                ).T + np.array([[s[0]], [s[1]]])
                pend = np.concatenate(
                    (
                        pend.T,
                        np.array([[0], [np.cos(s[5])], [np.cos(s[5])], [0]]),
                    ),
                    axis=1,
                )

                keypoints.append(pend)
            else:
                scale = R_psi @ np.array(
                    [
                        [-0.1, 0.4],
                        [0.1, 0.4],
                        [0.1, -0.4],
                        [-0.1, -0.4],
                    ]
                ).T + np.array([[s[0]], [s[1]]])

                keypoints.append(scale.T)

                d = 0.4 * 4 * s[5] / np.pi
                indicator = R_psi @ np.array(
                    [
                        [0, 0.1 + d],
                        [0.1, d],
                        [0, -0.1 + d],
                        [-0.1, d],
                    ]
                ).T + np.array([[s[0]], [s[1]]])

                keypoints.append(indicator.T)

        return keypoints


class Arrow2D:
    """A 2D arrow that may be drawn in the ground plane (z=0) of a 3D plot.

    DISCLAIMER: This class is under development and may not function properly.
    """

    def __init__(
        self, ax, x, y, dx, dy, headlength, headwidth, proj_3d=False, **kwargs
    ):
        """Draw a 2D arrow.

        The arrow will point from (x,y) to (x+dx, y+dy)

        Currently, the arrow cannot be animated.
        TODO: Add updateability to both 2D and 3D arrows to allow for
        animations.

        Parameters
        ----------
        ax : Axes
            Axes to be drawn in.
        x : float
            x-location of the arrow.
        y : TYPE
            y-location of the arrow.
        dx : float
            Width of the arrow.
        dy : float
            Height of the arrow.
        headlength : float
            Head length of the arrow as an absolute value.
        headwidth : float
            Head width of the arrow as an absolute value.
        proj_3d : bool, optional
            If True, project the arrow in the ground plane (z=0) of a 3D plot.
            The default is False.
        **kwargs
            Keyword arguments passed on to matplotlibs PolyCollection, Line2D,
            Line3D and Polygon. Use this to specify the style properties of
            the arrow. Only use keywords that are supported by all of the
            above.


        Returns
        -------
        None.

        """
        self.headlength = headlength
        self.headwidth = headwidth

        keypoints = self.calcKeypoints(x, y, dx, dy)

        if proj_3d:
            self.vect = Line3D(
                keypoints[0][:, 0],
                keypoints[0][:, 1],
                np.zeros_like(keypoints[0][:, 1]),
                **kwargs
            )
            self.head = PolyCollection((keypoints[1],), **kwargs)
        else:
            self.vect = Line2D(
                keypoints[0][:, 0], keypoints[0][:, 1], **kwargs
            )
            self.head = Polygon(keypoints[1], **kwargs)

        if proj_3d:
            ax.add_collection3d(self.head, zs=0)
        else:
            ax.add_patch(self.head)

        ax.add_artist(self.vect)

    def calcKeypoints(self, x, y, dx, dy):
        """Calculate the keypoints of the arrow.

        Parameters
        ----------
        x : float
            x-location of the arrow.
        y : TYPE
            y-location of the arrow.
        dx : float
            Width of the arrow.
        dy : float
            Height of the arrow.

        Returns
        -------
        xy_vect : Array
            Keypoints of the arrow tail.
        xy_head : Array
            Keypoints of the arrow head.

        """
        ang = np.arctan2(dy, dx)
        R = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])

        xy_head = np.array(
            [
                [0, -self.headlength, -self.headlength],
                [0, self.headwidth / 2, -self.headwidth / 2],
            ]
        )

        xy_head = R @ xy_head + np.array([[x + dx], [y + dy]])

        xy_vect = np.array([[x, y], [x + dx, y + dy]])

        return xy_vect, xy_head.T

    def update(self, x, y, dx, dy, headlength=None, headwidth=None, **kwargs):
        """Update the arrow locations and style.

        Parameters
        ----------
        ax : Axes
            Axes to be drawn in.
        x : float
            x-location of the arrow.
        y : TYPE
            y-location of the arrow.
        dx : float
            Width of the arrow.
        dy : float
            Height of the arrow.
        headlength : float
            Head length of the arrow as an absolute value.
        headwidth : float
            Head width of the arrow as an absolute value.
        **kwargs
            Keyword arguments passed on to matplotlibs PolyCollection, Line2D,
            Line3D and Polygon. Use this to specify the style properties of
            the arrow. Only use keywords that are supported by all of the
            above.

        Returns
        -------
        None.

        """
        if headlength is not None:
            self.headlength = headlength
        if headwidth is not None:
            self.headwidth = headwidth

        keypoints = self.calcKeypoints(x, y, dx, dy)

        self.vect.set_xy(keypoints[0])
        self.head.set_xy(keypoints[1])

        self.vect.set(**kwargs)
        self.head.set(**kwargs)
