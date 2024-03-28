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
            self.show_roll_indicator = (
                type(bike).__name__ == "InvPendulumBicycle"
            )

        if params is None:
            self.params = BikeDrawing2DParameters(
                show_roll_indicator=show_roll_indicator, proj_3d=proj_3d
            )
        else:
            self.params = params
            if not show_roll_indicator:
                self.params.show_roll_indicator = False
                self.params.make_colorlists_riderbike()
            if proj_3d:
                self.params.proj_3d = True
                self.params.make_colorlists_riderbike()

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

    def make_force_arrows(self):
        """Create a set of lists to storing the handles of force arrows

        Parameters
        ----------
        None.

        Returns
        -------
        None.

        """

        self.force_handle_dest = self.ax.arrow(
            0,
            0,
            0,
            0,
            head_width=0.3,
            head_length=0.4,
            linewidth=1,
            edgecolor="gray",
            facecolor="gray",
            animated=True,
            zorder=3,
        )

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
