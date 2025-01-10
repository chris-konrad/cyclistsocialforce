# -*- coding: utf-8 -*-
"""
Created on Tue Sep 26 12:16:53 2023.

Classes managing the parameter sets of the model and the visualisation.

@author: Christoph M. Schmidt
"""

import numpy as np
import control as ct
import warnings

from cyclistsocialforce.utils import thresh
from pypaperutils.design import TUDcolors

from bicycleparameters.parameter_dicts import meijaard2007_browser_jason
from bicycleparameters.parameter_sets import Meijaard2007ParameterSet
from bicycleparameters.models import Meijaard2007Model


class VehicleDrawingParameters:
    """Class storing and maintaining the parameters for a vehicle drawing.

    Parameters include colors,
    To be used together with cyclistsocialforce.visualisation.Vehicle
    """

    def __init__(
        self,
        animated=False,
        draw_force_resulting=True,
        draw_force_destination=True,
        draw_forces_repulsive=True,
        draw_trajectory=True,
        draw_nextdest=False,
        draw_destqueue=True,
        draw_pastdest=True,
        draw_name=True,
        force_color_dest=None,
        force_color_rep=None,
        force_color_res=None,
        force_head_width=None,
        force_head_length=None,
        force_linewidth=None,
        dest_marker_color_cur=None,
        dest_marker_color_qeu=None,
        traj_line_width=None,
        traj_line_color=None,
        name_font_size=None,
        name_font_color=None,
    ):
        self.draw_force_resulting = draw_force_resulting
        self.draw_force_destination = draw_force_destination
        self.draw_forces_repulsive = draw_forces_repulsive
        self.draw_trajectory = draw_trajectory
        self.draw_nextdest = draw_nextdest
        self.draw_destqueue = draw_destqueue
        self.draw_pastdest = draw_pastdest
        self.draw_name = draw_name
        self.animated = animated

        self.tud_colors = TUDcolors()

        self.init_forcearrow_style(
            force_color_dest, force_color_rep, force_color_res, force_head_width, force_head_length,
            force_linewidth
        )
        
        self.init_destmarker_colors(dest_marker_color_cur, 
                                    dest_marker_color_qeu)
        
        self.init_trajectory_style(traj_line_width, traj_line_color)
        
        self.init_name_style(name_font_size, name_font_color)
        
    def init_name_style(self, name_font_size=None, name_font_color=None):
        
        if name_font_size is None:
            name_font_size = 8
        if name_font_color is None:
            name_font_color = "black"

        self.name_font_size = name_font_size
        self.name_font_color = name_font_color
        
    def init_trajectory_style(
        self, traj_line_width=None, traj_line_color=None
        ):
        
        if traj_line_width is None:
            traj_line_width = 1
        if traj_line_color is None:
            traj_line_color = self.tud_colors.get("cyaan")

        self.traj_line_width = traj_line_width
        self.traj_line_color = traj_line_color
        
    def init_destmarker_colors(
        self, dest_marker_color_cur=None, dest_marker_color_qeu=None
    ):
        """Initializes the marker colors of the destination markers.
        

        Parameters
        ----------
        dest_marker_color_cur : color, optional
            The default is gray.
        dest_marker_color_qeu : color, optional
            The default is gray.

        Returns
        -------
        None.

        """
        if dest_marker_color_cur is None:
            dest_marker_color_cur = "gray"
        if dest_marker_color_qeu is None:
            dest_marker_color_qeu = "gray"

        self.dest_marker_color_cur = dest_marker_color_cur
        self.dest_marker_color_qeu = dest_marker_color_qeu

    def init_forcearrow_style(
        self, force_color_dest=None, force_color_rep=None, force_color_res=None,
        force_head_width=None, force_head_length=None, force_linewidth=None
    ):
        """Initializes the face and edge colors for the force arrows.

        Parameters
        ----------
        force_color_dest : color, optional
            The default is gray.
        force_color_rep : color, optional
            The default is gray.
        force_color_res : color, optional
            The default is something dark.
        force_head_width : float, optional
            The default is 0.4
        force_head_length : float, optional
            The default is 0.4
        force_linewidth : float, optional
            The default is 1.0
        

        Returns
        -------
        None.

        """
        if force_color_dest is None:
            force_color_dest = "gray"
        if force_color_rep is None:
            force_color_rep = "gray"
        if force_color_res is None:
            force_color_res = (12.0 / 255, 35.0 / 255, 64.0 / 255)
        if force_head_width is None:
            force_head_width = 0.3
        if force_head_length is None:
            force_head_length = 0.4
        if force_linewidth is None:
            force_linewidth = 1.0           

        self.force_color_dest = force_color_dest
        self.force_color_rep = force_color_rep
        self.force_color_res = force_color_res
        self.force_head_length = force_head_length
        self.force_head_width = force_head_width
        self.force_linewidth = force_linewidth

    def get_draw_forces(self):
        return (
            self.draw_forces_destination
            or self.draw_forces_repulsive
            or self.draw_forces_resulting
        )


class BikeDrawing2DParameters(VehicleDrawingParameters):
    """Class storing and maintaining the parameters for a bicycle drawing.

    Parameters include colors,
    To be used together with cyclistsocialforce.visualisation.BicycleDrawing2D

    """

    def __init__(
        self,
        bike_color_frame=None,
        bike_color_wheels=None,
        rider_color_body=None,
        rider_color_head=None,
        roll_indicator_color_edge=None,
        roll_indicator_color_bg=None,
        roll_indicator_color_marker=None,
        draw_roll_indicator=True,
        proj_3d=False,
        **kwargs,
    ):
        """Create a bicycle drawing parameters object.


        Parameters
        ----------
        bike_color_frame : color, optional
            The default is TU Delft cyan.
        bike_color_wheels : color, optional
            The default is gray.
        rider_color_body : color or list of colors, optional
            The default is random sampling from all TU Delft colors. If
            a list of colors is provided the body color is randomly sampled
            from this list.
        rider_color_head : color, optional
            The default is TU Delft cyan.
        roll_indicator_color_edge : color, optional
            The default is black,
        roll_indicator_color_bg : color, optional
            The default is None (transparent).
        roll_indicator_color_marker : color, optional
            The default is red.
        draw_roll_indicator : boolean, optional
            Adds the roll indicator colors to the color lists.
            The default is True.
        proj3d : TYPE, optional
            Prepares color lists for the 3D drawing instead of 2D.
            The default is False.

        Returns
        -------
        None.

        """
        super().__init__(**kwargs)

        self.proj_3d = proj_3d
        self.draw_roll_indicator = draw_roll_indicator

        self.init_riderbike_colors(
            bike_color_frame,
            bike_color_wheels,
            rider_color_body,
            rider_color_head,
            roll_indicator_color_edge,
            roll_indicator_color_bg,
            roll_indicator_color_marker,
        )
        self.make_colorlists_riderbike()

    def init_riderbike_colors(
        self,
        bike_color_frame=None,
        bike_color_wheels=None,
        rider_color_body=None,
        rider_color_head=None,
        roll_indicator_color_edge=None,
        roll_indicator_color_bg=None,
        roll_indicator_color_marker=None,
    ):
        """Initializes the face and edge colors for the bike-rider polygon
        including the roll indicator.


        Parameters
        ----------
        bike_color_frame : color, optional
            The default is TU Delft cyan.
        bike_color_wheels : color, optional
            The default is gray.
        rider_color_body : color or list of colors, optional
            The default is random sampling from all TU Delft colors. If
            a list of colors is provided the body color is randomly sampled
            from this list.
        rider_color_head : color, optional
            The default is TU Delft cyan.
        roll_indicator_color_edge : color, optional
            The default is black,
        roll_indicator_color_bg : color, optional
            The default is None (transparent).
        roll_indicator_color_marker : color, optional
            The default is red.

        Returns
        -------
        None.

        """

        if bike_color_frame is None:
            bike_color_frame = self.tud_colors.get("cyaan")
        if bike_color_wheels is None:
            bike_color_wheels = "gray"

        if rider_color_body is None:
            rider_color_body = self.tud_colors.get(
                np.random.randint(0, len(self.tud_colors.colors))
            )
        elif isinstance(rider_color_body, list):
            rider_color_body = rider_color_body[
                np.random.randint(0, len(rider_color_body))
            ]

        if rider_color_head is None:
            rider_color_head = self.tud_colors.get("cyaan")

        if roll_indicator_color_edge is None:
            roll_indicator_color_edge = "black"
        if roll_indicator_color_bg is None:
            roll_indicator_color_bg = "none"
        if roll_indicator_color_marker is None:
            roll_indicator_color_marker = self.tud_colors.get("rood")

        self.bike_color_frame = bike_color_frame
        self.bike_color_wheels = bike_color_wheels
        self.rider_color_body = rider_color_body
        self.rider_color_head = rider_color_head
        self.roll_indicator_color_edge = roll_indicator_color_edge
        self.roll_indicator_color_bg = roll_indicator_color_bg
        self.roll_indicator_color_marker = roll_indicator_color_marker

    def make_colorlists_riderbike(self):
        """Create the list of colors for the rider+bike polygons.


        Returns
        -------
        None.

        """

        self.fcolors_riderbike = [
            self.bike_color_wheels,
            self.bike_color_wheels,
            self.bike_color_frame,
            self.bike_color_frame,
            self.rider_color_body,
            self.rider_color_body,
            self.rider_color_body,
            self.rider_color_head,
        ]

        self.ecolors_riderbike = ["none"] * 8

        if self.draw_roll_indicator:
            if self.proj_3d:
                self.fcolors_riderbike += [
                    self.roll_indicator_color_edge,
                ]
                self.ecolors_riderbike += [
                    "none",
                ]
            else:
                self.fcolors_riderbike += [
                    self.roll_indicator_color_bg,
                    self.roll_indicator_color_marker,
                ]
                self.ecolors_riderbike += [
                    self.roll_indicator_color_edge,
                    "none",
                ]


class RoadElementParameters:
    def __init__(
        self,
        roadsurface_color=(0.8, 0.8, 0.8),
        roadedge_color="white",
        roadedge_linewidth=1,
        F_0=0.05,
        sigma=3.0,
    ):
        self.roadsurface_color = roadsurface_color

        self.roadedge_color = roadedge_color
        self.roadedge_linewidth = roadedge_linewidth

        self.F_0 = F_0
        self.sigma = sigma

    @property
    def F_0(self) -> float:
        return self._F_0

    @F_0.setter
    def F_0(self, F_0) -> None:
        if hasattr(self, "_F_0"):
            raise AttributeError("F_0 is immutable.")
        if not isinstance(F_0, float):
            msg = "F_0 must be a float."
            raise TypeError(msg)
        if not F_0 >= 0:
            raise ValueError(
                f"F_0 must be >=0, instead it was \
                             {F_0:.2f}"
            )
        self._F_0 = F_0

    @property
    def sigma(self) -> float:
        return self._sigma

    @sigma.setter
    def sigma(self, sigma) -> None:
        if hasattr(self, "_sigma"):
            raise AttributeError("sigma is immutable.")
        if not isinstance(sigma, float):
            msg = "sigma must be a float."
            raise TypeError(msg)
        if not sigma >= 0:
            raise ValueError(
                f"sigma must be >=0, instead it was \
                             {sigma:.2f}"
            )
        self._sigma = sigma


class VehicleParameters:
    """Base class for all Vehicle-specific paramter classes.
    Calculate and update the parameters of a vehicle.

    Provides tactical parameters.
    """

    LIMIT_PREC = 1e-4

    def __init__(
        self,
        t_s: float = 0.01,
        d_arrived_inter: float = 2.0,
        d_arrived_stop: float = 2.0,
        v_max_stop: float = 0.1,
        v_max_harddecel: float = 2.5,
        hfov: float = 2 * np.pi,
        calib_mode=False,
        verbose=True,
        rep_force={},
        dest_force={},
        dynamics={},
        # repulsive force field parameters
        f_0: float = 7.0,
        e_0: float = 0.995,
        e_1: float = 0.7,
        sigma_0: float = 0.5,
        sigma_1: float = 5.0,
        sigma_2: float = 0.3,
        sigma_3: float = 4.9,
    ) -> None:
        """Create the parameter set of a default vehicle.

        TODO:
        - Add interface to set parameters to values other then the default.
        - Add randomness.
        - Add compatibility to SUMO *.rou.xml.


        parameters
        ----------
        t_s : float, optional
            Sample time. The default is 0.01.
        d_arrived_inter : float, optional
            Distance to intermediate destination at which the destination
            counts are "arrived" and the vehicle continues to the next
            destination. The default is 2.0.
        d_arrived_stop : float, optional
            Distance to stop destination at which the destination counts as
            "arrived" and the vehicle may stop. The default is 2.0.
        v_max_stop : float, optional
            Maximum speed below which the vehcile may count as stopped.
            The default is 0.1.
        v_max_harddecel : float, optional
            Maximum speed below which the vehicle may apply hard decceleration
            for a normal stop at a stop destination. The default is 2.5.
        hfov : float, optional
            Horizontal field of view of the vehicle driver/rider. The vehicle
            will react to other road users within that field of view and ignore
            those outside. The default is 2*np.pi.

        Returns
        -------
        None


        """
        self.calib_mode = calib_mode
        self.verbose = verbose

        self.t_s = t_s
        self.d_arrived_inter = d_arrived_inter
        self.d_arrived_stop = d_arrived_stop
        self.v_max_stop = v_max_stop
        self.v_max_harddecel = v_max_harddecel
        self.hfov = hfov
        
        self.rep_force = rep_force
        self.dest_force = dest_force

        self._e_1 = 0  # set this before the first e_0 assignment, otherwise the value check of e_0 does not work.
        self.f_0 = f_0
        self.e_0 = e_0
        self.e_1 = e_1
        self.sigma_0 = sigma_0
        self.sigma_1 = sigma_1
        self.sigma_2 = sigma_2
        self.sigma_3 = sigma_3

    # ---- PROPERTIES ----

    @property
    def t_s(self) -> float:
        return self._t_s

    @t_s.setter
    def t_s(self, t_s) -> None:
        if hasattr(self, "_t_s"):
            raise AttributeError("t_s is immutable.")
        if not isinstance(t_s, float):
            msg = "t_s must be a float."
            raise TypeError(msg)
        if not t_s >= 0:
            raise ValueError(
                f"t_s must be >=0, instead it was \
                             {t_s:.2f}"
            )
        self._t_s = t_s

    @property
    def d_arrived_inter(self) -> float:
        return self._d_arrived_inter

    @d_arrived_inter.setter
    def d_arrived_inter(self, d_arrived_inter) -> None:
        if not isinstance(d_arrived_inter, float):
            raise TypeError("d_arrived_inter must be a float")
        if not d_arrived_inter >= 0:
            raise ValueError(
                f"d_arrived_inter must be >=0, instead it was \
                             {d_arrived_inter:.2f}"
            )
        self._d_arrived_inter = d_arrived_inter

    @property
    def d_arrived_stop(self) -> float:
        return self._d_arrived_stop

    @d_arrived_stop.setter
    def d_arrived_stop(self, d_arrived_stop) -> None:
        if hasattr(self, "_d_arrived_stop"):
            raise AttributeError("d_arrived_stop is immutable.")
        if not isinstance(d_arrived_stop, float):
            msg = "d_arrived_stop must be a float."
            raise TypeError(msg)
        if not d_arrived_stop >= 0:
            raise ValueError(
                f"d_arrived_stop must be >=0, instead it was \
                             {d_arrived_stop:.2f}"
            )
        self._d_arrived_stop = d_arrived_stop

    @property
    def v_max_stop(self) -> float:
        return self._v_max_stop

    @v_max_stop.setter
    def v_max_stop(self, v_max_stop) -> None:
        if hasattr(self, "_v_max_stop"):
            raise AttributeError("v_max_stop is immutable.")
        if not isinstance(v_max_stop, float):
            msg = "v_max_stop must be a float."
            raise TypeError(msg)
        if not v_max_stop >= 0:
            raise ValueError(
                f"v_max_stop must be >=0, instead it was \
                             {v_max_stop:.2f}"
            )
        self._v_max_stop = v_max_stop

    @property
    def v_max_harddecel(self) -> float:
        return self._v_max_harddecel

    @v_max_harddecel.setter
    def v_max_harddecel(self, v_max_harddecel) -> None:
        if hasattr(self, "_v_max_harddecel"):
            raise AttributeError("v_max_harddecel is immutable.")
        if not isinstance(v_max_harddecel, float):
            msg = "v_max_harddecel must be a float."
            raise TypeError(msg)
        if not v_max_harddecel >= 0:
            raise ValueError(
                f"v_max_harddecel must be >=0, instead it was \
                             {v_max_harddecel:.2f}"
            )
        self._v_max_harddecel = v_max_harddecel

    @property
    def hfov(self) -> float:
        return self._hfov

    @hfov.setter
    def hfov(self, hfov) -> None:
        if hasattr(self, "_hfov"):
            raise AttributeError("hfov is immutable.")
        if not isinstance(hfov, float):
            msg = "hfov must be a float."
            raise TypeError(msg)
        if not 0 < hfov <= 2 * np.pi:
            raise ValueError(
                f"hfov must be in ]0,2*pi], instead it was \
                             {hfov:.2f}"
            )
        self._hfov = hfov

    @property
    def f_0(self) -> float:
        return self._f_0

    @f_0.setter
    def f_0(self, f_0) -> None:
        f_0 = float(f_0)
        if not f_0 >= 0:
            msg = f"f_0 must be >=0, instead it was {f_0:.2f}"
            if self.calib_mode:
                if self.verbose: warnings.warn(msg)
                f_0 = self.LIMIT_PREC
            else:
                raise ValueError(msg)
        self._f_0 = f_0

    @property
    def e_0(self) -> float:
        return self._e_0

    @e_0.setter
    def e_0(self, e_0) -> None:
        if not isinstance(e_0, float):
            raise TypeError("e_0 must be a float.")
        if not self.e_1 < e_0 <= 1:
            msg = f"e_0 must be in ]e_1={self.e_1:.2f}, 1], instead it was {e_0:.2f}"
            if self.calib_mode:
                if self.verbose: warnings.warn(msg)
                e_0 = thresh(e_0, (self.e_1 * 1.001, 0.99999))
            else:
                raise ValueError(msg)
        self._e_0 = e_0

    @property
    def e_1(self) -> float:
        return self._e_1

    @e_1.setter
    def e_1(self, e_1) -> None:
        if not isinstance(e_1, float):
            raise TypeError("e_1 must be a float.")
        if not 0 <= e_1 < self.e_0:
            msg = f"e_1 must be in [0,e_0={self.e_0:.2f}[, instead it was {e_1:.2f}"
            if self.calib_mode:
                if self.verbose: warnings.warn(msg)
                e_1 = thresh(e_1, (0, 0.99999 * self.e_0))
            else:
                raise ValueError(msg)
        self._e_1 = e_1

    @property
    def sigma_0(self) -> float:
        return self._sigma_0

    @sigma_0.setter
    def sigma_0(self, sigma_0) -> None:
        if not isinstance(sigma_0, float):
            raise TypeError("sigma_0 must be a float.")
        if not sigma_0 >= 0:
            msg = f"sigma_0 must be >=0, instead it was {sigma_0:.2f}"
            if self.calib_mode:
                if self.verbose: warnings.warn(msg)
                sigma_0 = self.LIMIT_PREC
            else:
                raise ValueError(msg)
        self._sigma_0 = sigma_0

    @property
    def sigma_1(self) -> float:
        return self._sigma_1

    @sigma_1.setter
    def sigma_1(self, sigma_1) -> None:
        if not isinstance(sigma_1, float):
            raise TypeError("sigma_1 must be a float.")
        if not sigma_1 >= 0:
            msg = f"sigma_1 must be >=0, instead it was {sigma_1:.2f}"
            if self.calib_mode:
                if self.verbose: warnings.warn(msg)
                sigma_1 = self.LIMIT_PREC
            else:
                raise ValueError(msg)
        self._sigma_1 = sigma_1

    @property
    def sigma_2(self) -> float:
        return self._sigma_2

    @sigma_2.setter
    def sigma_2(self, sigma_2) -> None:
        if not isinstance(sigma_2, float):
            raise TypeError("sigma_2 must be a float.")
        if not 0 < sigma_2 < self.sigma_0:
            msg = f"sigma_2 must be in [0,sigma_0={self.sigma_0:.2f}[, instead it was {sigma_2:.2f}"
            if self.calib_mode:
                if self.verbose: warnings.warn(msg)
                sigma_2 = thresh(sigma_2, (0, self.sigma_0 - self.LIMIT_PREC))
            else:
                raise ValueError(msg)
        self._sigma_2 = sigma_2

    @property
    def sigma_3(self) -> float:
        return self._sigma_3

    @sigma_3.setter
    def sigma_3(self, sigma_3) -> None:
        if not isinstance(sigma_3, float):
            raise TypeError("sigma_3 must be a float.")
        if not 0 < sigma_3 < self.sigma_1:
            msg = f"sigma_3 must be in [0,sigma_1={self.sigma_1:.2f}[, instead it was {sigma_3:.2f}"
            if self.calib_mode:
                if self.verbose: warnings.warn(msg)
                sigma_0 = thresh(sigma_3, (0, self.sigma_1 - self.LIMIT_PREC))
            else:
                raise ValueError(msg)
        self._sigma_3 = sigma_3

    # ---- METHODS ----

    def __str__(self):
        """Create string with a list of properties and values.

        Returns
        -------
        s : str
            Property-value list.

        """
        vardict = vars(self)
        s = ""
        for key in vardict:
            s += key + f" : {vardict[key]}\n"
        return s


class CarParameters(VehicleParameters):
    def __init__(
        self,
        length=4,
        width=2.0,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.length = length
        self.width = width


class BicycleParameters(VehicleParameters):
    """Calculate and update the parameters of a bicycle and it's rider.

    Inherits tactical parameters from VehicleParameters.

    Provides dynamic and social force parameters common to all bicycles.
    Provides physical and control parameters for the stable bicycle.

    Default pysical bike parameter values taken from:
        Moore, J. K.(2015, June 30). Bicycle Control Design in Python/v3.
        Plotly Graphing Libraries. https://plotly.com/python/v3/ipython-
        notebooks/bicycle-control-design/
    """

    def __init__(
        self,
        v_max_riding: tuple = [-1.0, 10.0],
        v_desired_default: float = 5.0,
        p_decay: float = 5.0,
        p_0: float = 30.0,
        hfov: float = np.pi * 2 / 3,
        v_max_stop: float = 0.6,
        l: float = 1.0,
        l_1: float = None,
        l_2: float = None,
        delta_max: float = 1.4,
        a_max: tuple = [-10.0, 10.0],
        a_desired_default: tuple = [-5.0, 5.0],
        k_p_v: float = 10.0,
        k_p_delta: float = 10.0,
        t_s: float = 0.01,
        d_arrived_inter: float = 2.0,
        d_arrived_stop: float = 2.0,
        v_max_harddecel: float = 2.5,
        g = 9.81,
        **kwargs,
    ) -> None:
        """Create the parameter set of a default bicycle.

        TODO:
        - Add interface to set parameters to values other then the
        default.
        - Add randomness.
        - Add compatibility to SUMO *.rou.xml.


        Parameters
        ----------
        t_s : float, optional
            Sample time. The default is 0.01.
        d_arrived_inter : float, optional
            Distance to intermediate destination at which the destination
            counts are "arrived" and the vehicle continues to the next
            destination. The default is 2.0.
        d_arrived_stop : float, optional
            Distance to stop destination at which the destination counts as
            "arrived" and the vehicle may stop. The default is 2.0.
        v_max_stop : float, optional
            Maximum speed below which the vehcile may count as stopped.
            The default is 0.1.
        v_max_harddecel : float, optional
            Maximum speed below which the vehicle may apply hard decceleration
            for a normal stop at a stop destination. The default is 2.5.
        v_max_riding : tuple, optional
            Maximal longitudinal velocity of the bicycle for forward and back-
            ward motion, given as [backward, forward]. The default is [-1.,7.].
        v_desired_default : float, optional
            Default desired forward velocity of the cyclist. The default is 5..
        p_decay : float, optional
            Repulisve force potential decay. The default is 5..
        p_0 : float, optional
            Repulsive force potenital magnitude at r=0. The default is 30..
        hfov : float, optional
            Horizontal field of view of the cyclist. The vehicle will react to
            other road users within that field of view and ignore those
            outside. The default is np.pi*2/3.
        l : float, optional
            Wheelbase of the bicycle. The default is 1.0. based on
            Moore (2015).
        l_1 : float, optional
            Front section of the wheelbase of the bicycle. The default is 0.5,
            based on Moore (2015).
        l_2 : float, optional
            Front section of the wheelbase of the bicycle. The default is 0.5,
            based on Moore (2015).
        delta_max : float, optional
            Mechanical maximum of the steering angle (symmetric). The default
            is 1.4.
        a_max : tuple, optional
            Maximum possible de-/acceleration range given as [breaking,
            accelerating]. The default is (-10.,10.).
        a_desired_default : tuple, optional
            Default desired de-/acceleration range given as [breaking,
            accelerating]. The default is (-5., 5.).
        k_p_v : float, optional
            Proportional gain for velocity control. The default is 10.0.
        k_p_delta : float, optional
            Proportional gain for steer angle control. The default is 10.0.

        Returns
        -------
        None

        """

        VehicleParameters.__init__(
            self,
            t_s=t_s,
            d_arrived_inter=d_arrived_inter,
            d_arrived_stop=d_arrived_stop,
            v_max_stop=v_max_stop,
            v_max_harddecel=v_max_harddecel,
            hfov=hfov,
            **kwargs,
        )

        # Dynamic v_max_ridingter default values
        self.v_max_riding = v_max_riding
        self.v_desired_default = v_desired_default

        # Social force parameter defauls
        self.p_decay = p_decay
        self.p_0 = p_0

        # Pysical bike parameter default value
        if l_1 is None and l_2 is None:
            assert l is not None, "If l_1 and l_2 are None, l may not be None!"
            l_1 = l / 2
            l_2 = l / 2
        if l is None:
            assert l_1 is not None, "Only one of l, l_1, l_2 may be None!"
            assert l_2 is not None, "Only one of l, l_1, l_2 may be None!"
            self.l_1 = l_1
            self.l_2 = l_2
            self.l = None
        elif l_1 is None:
            assert l is not None, "Only one of l, l_1, l_2 may be None!"
            assert l_2 is not None, "Only one of l, l_1, l_2 may be None!"
            self.l_2 = l_2
            self.l = l
            self.l_1 = None
        elif l_2 is None:
            assert l is not None, "Only one of l, l_1, l_2 may be None!"
            assert l_1 is not None, "Only one of l, l_1, l_2 may be None!"
            self.l_1 = l_1
            self.l = l
            self.l_2 = None
        else:
            assert l == l_1 + l_2, (
                "Equality l = l_1 + l_2 must hold! Set "
                "one of l, l_1, l_2 to None to automatically calculate the "
                "last parameter and ensure equality."
            )
            self.l = l
            self.l_1 = l_1
            self.l_2 = l_2

        self.delta_max = delta_max

        # Dynamic bike parameters
        self.a_max = a_max
        self.a_desired_default = a_desired_default

        # Control parameter default values
        self.k_p_v = k_p_v
        self.k_p_delta = k_p_delta
        
        #Physical constants
        self.g = g
 
    # ---- PROPERTIES ----

    @property
    def v_max_riding(self) -> float:
        return self._v_max_riding

    @v_max_riding.setter
    def v_max_riding(self, v_max_riding) -> None:
        if hasattr(self, "_v_max_riding"):
            raise AttributeError("v_max_riding is immutable.")
        if not isinstance(v_max_riding, (list, tuple)):
            raise TypeError("v_max_riding must be list or tuple.")
        if (not isinstance(v_max_riding[0], float)) or (
            not isinstance(v_max_riding[1], float)
        ):
            raise TypeError(
                "v_max_riding[0] and v_max_riding[1] \
                            must be float."
            )
        if (not v_max_riding[1] > 0) or (not v_max_riding[0] < 0):
            raise ValueError(
                f"v_max_riding[0] must be <0 and \
                             v_max_riding[1] must be >0, instead it was \
                             {v_max_riding}"
            )
        self._v_max_riding = v_max_riding

    @property
    def v_desired_default(self) -> float:
        return self._v_desired_default

    @v_desired_default.setter
    def v_desired_default(self, v_desired_default) -> None:
        if not isinstance(v_desired_default, float):
            msg = "v_desired_default must be a float."
            raise TypeError(msg)
        if not v_desired_default >= 0:
            raise ValueError(
                f"v_desired_default must be >=0, instead it was \
                             {v_desired_default:.2f}"
            )
        self._v_desired_default = v_desired_default

    @property
    def p_decay(self) -> float:
        return self._p_decay

    @p_decay.setter
    def p_decay(self, p_decay) -> None:
        if hasattr(self, "_p_decay"):
            raise AttributeError("p_decay is immutable.")
        if not isinstance(p_decay, float):
            msg = "p_decay must be a float."
            raise TypeError(msg)
        if not p_decay >= 0:
            raise ValueError(
                f"p_decay must be >=0, instead it was \
                             {p_decay:.2f}"
            )
        self._p_decay = p_decay

    @property
    def p_0(self) -> float:
        return self._p_0

    @p_0.setter
    def p_0(self, p_0) -> None:
        if hasattr(self, "_p_0"):
            raise AttributeError("p_0 is immutable.")
        if not isinstance(p_0, float):
            msg = "p_0 must be a float."
            raise TypeError(msg)
        if not p_0 >= 0:
            raise ValueError(
                f"p_0 must be >=0, instead it was \
                             {p_0:.2f}"
            )
        self._p_0 = p_0

    @property
    def l(self) -> float:
        return self._l

    @l.setter
    def l(self, l) -> None:
        if hasattr(self, "_l"):
            raise AttributeError("l is immutable.")
        if l is None:
            self._l = self.l_1 + self.l_2
        else:
            if not isinstance(l, float):
                msg = "l must be a float."
                raise TypeError(msg)
            if not l >= 0:
                raise ValueError(
                    f"l must be >=0, instead it was \
                                 {l:.2f}"
                )
            self._l = l

    @property
    def l_1(self) -> float:
        return self._l_1

    @l_1.setter
    def l_1(self, l_1) -> None:
        if hasattr(self, "_l_1"):
            raise AttributeError("l_1 is immutable.")
        if l_1 is None:
            self._l_1 = self.l - self._l_2
        else:
            if not isinstance(l_1, float):
                raise TypeError("l_1 must be a float.")
            if not l_1 >= 0:
                raise ValueError(
                    f"l_1 must be >=0, instead it was \
                                 {l_1:.2f}"
                )
            self._l_1 = l_1

    @property
    def l_2(self) -> float:
        return self._l_2

    @l_2.setter
    def l_2(self, l_2) -> None:
        if hasattr(self, "_l_2"):
            raise AttributeError("l_2 is immutable.")
        if l_2 is None:
            self._l_2 = self.l - self._l_1
        else:
            if not isinstance(l_2, float):
                raise TypeError("l_2 must be a float.")
            if not l_2 >= 0:
                raise ValueError(
                    f"l_2 must be >=0, instead it was \
                                 {l_2:.2f}"
                )
            self._l_2 = l_2

    @property
    def delta_max(self) -> float:
        return self._delta_max

    @delta_max.setter
    def delta_max(self, delta_max) -> None:
        if hasattr(self, "_delta_max"):
            raise AttributeError("delta_max is immutable.")
        if not isinstance(delta_max, float):
            msg = "delta_max must be a float."
            raise TypeError(msg)
        if not 0 <= delta_max <= np.pi:
            raise ValueError(
                f"delta_max must be in [0,pi], instead it was \
                             {delta_max:.2f}"
            )
        self._delta_max = delta_max

    @property
    def a_max(self) -> float:
        return self._a_max

    @a_max.setter
    def a_max(self, a_max) -> None:
        if hasattr(self, "_a_max"):
            raise AttributeError("a_max is immutable.")
        if not isinstance(a_max, (tuple, list)):
            raise TypeError("a_max must be a tuple or list.")
        if (not isinstance(a_max[0], float)) or (
            not isinstance(a_max[1], float)
        ):
            raise TypeError("a_max[0] and a_max[1] must be float.")
        if (not a_max[1] > 0) or (not a_max[0] < 0):
            raise ValueError(
                f"a_max[0] must be <0 and \
                             a_max[1] must be >0, instead it was \
                             {a_max}"
            )
        self._a_max = a_max

    @property
    def a_desired_default(self) -> float:
        return self._a_desired_default

    @a_desired_default.setter
    def a_desired_default(self, a_desired_default) -> None:
        if hasattr(self, "_a_desired_default"):
            raise AttributeError("a_desired_default is immutable.")
        if not isinstance(a_desired_default, (list, tuple)):
            raise TypeError("a_desired_default must be list or tuple.")
        if (not isinstance(a_desired_default[0], float)) or (
            not isinstance(a_desired_default[1], float)
        ):
            raise TypeError(
                "a_desired_default[0] and a_desired_default[1] \
                            must be float."
            )
        if (not a_desired_default[1] > 0) or (not a_desired_default[0] < 0):
            raise ValueError(
                f"a_desired_default[0] must be <0 and \
                             a_desired_default[1] must be >0, instead it was \
                             {a_desired_default}"
            )
        self._a_desired_default = a_desired_default

    @property
    def k_p_v(self) -> float:
        return self._k_p_v

    @k_p_v.setter
    def k_p_v(self, k_p_v) -> None:
        if hasattr(self, "_k_p_v"):
            raise AttributeError("k_p_v is immutable.")
        if not isinstance(k_p_v, float):
            raise TypeError("k_p_v must be a float.")
        if not k_p_v >= 0:
            raise ValueError(
                f"k_p_v must be >=0, instead it was \
                             {k_p_v:.2f}"
            )
        self._k_p_v = k_p_v

    @property
    def k_p_delta(self) -> float:
        return self._k_p_delta

    @k_p_delta.setter
    def k_p_delta(self, k_p_delta) -> None:
        if hasattr(self, "_k_p_delta"):
            raise AttributeError("k_p_delta is immutable.")
        if not isinstance(k_p_delta, float):
            raise TypeError("k_p_delta must be a float.")
        if not k_p_delta >= 0:
            raise ValueError(
                f"k_p_delta must be >=0, instead it was \
                             {k_p_delta:.2f}"
            )
        self._k_p_delta = k_p_delta
        
class ParticleBicycleParameters(BicycleParameters):
    
    FIXED_POLES = 0+0j #must have a double pole at FIXED_POLES
    N_POLES = 4
    
    def __init__(self, 
                 poles = None, 
                 gains = None,
                 **kwargs):
        BicycleParameters.__init__(self, **kwargs)
        self.gains = gains
        self.poles = poles
        
    @property
    def poles(self):
        return self._poles
    @poles.setter
    def poles(self, poles):
        if poles is None:
            poles = (0 + 0j, 
                     0 + 0j, 
                     -0.97626953125+0j + 0j, 
                     -0.97626953125+0j + 0j)
        if len(poles) != 4:
            msg = "ParticleBicycleParameters must have four poles! Instead" \
                  f"you provided {len(poles)}: poles = {poles}"
            ValueError(msg)
        if np.where(poles == self.FIXED_POLES)[0].size != 2:
            msg = "ParticleBicycleParameters must have a double pole at " \
                  f"{self.FIXED_POLES}. Instead the given poles where: " \
                  f"poles = {poles}"
        self._poles = poles
        
class PlanarBicycleParameters(BicycleParameters):
    
    def __init__(self, 
                 poles = (-1.0141284591434665 + 1.226826644413086j,
                          -1.0141284591434665 - 1.226826644413086j),
                 **kwargs):
        
        BicycleParameters.__init__(self, **kwargs)
        self.poles = poles    

        
class WhippleCarvalloBicycleParameters(BicycleParameters):
    
    def __init__(self, 
                 bicycleParameterDict = meijaard2007_browser_jason, 
                 poles = (-18 + 0j, -19 + 0, -20 + 0j, 
                          -1.2240726404163056+1.2879634520488243j, 
                          -1.2240726404163056-1.2879634520488243j),
                 p_dist_roll = 0.00,
                 p_dist_steer = 0.00,
                 T_dist_roll = 9000,
                 T_dist_steer = 1000,
                 gains = None,
                 **kwargs):
        """
        Generate a parameters object for the Whipple-Carvallo Bicycle.

        Parameters
        ----------
        bicycleParameterDict : dict, optional
            Dictionary with bicycle parameters from 
            bicycleparameters.parameter_dicts. The default is 
            bicycleparameters.parameter_dicts.meijaard2007_browser_jason.
        poles : array-like, optional
            Pole locations of the Full-state-feedback Whipple-Carvallo model. 
            The default is (-18 + 0j, -19 + 0, -20 + 0j, -1.2240726404163056+
            1.2879634520488243j, -1.2240726404163056-1.2879634520488243j), 
            which was obtained from pilot calibration tests. 
        p_dist_roll : float, optional
            Probability p of a roll torque disturbance occuring at any given 
            time step. The default is 0.00.
        p_dist_steer : float, optional
            Probability p of a steer torque disturbance occuring at any given 
            time step. The default is 0.00.
        T_dist_roll : float, optional
            Magnitude of the disturbance roll torque. The default is 9000 N.
        T_dist_steer : float, optional
            Magnitude of the disturbance steer torque. The default is 1000 N.
        **kwargs 
            Keyword argumens of BicycleParameters.

        Returns
        -------
        None.

        """
        
        # Meijard(2007) model and parameter set from bicycleparameters
        p = Meijaard2007ParameterSet(bicycleParameterDict, True)
        m = Meijaard2007Model(p)
        
        #set wheelbase to the parameters in the parameter dict. This overwrites
        #any manually given wheelbase. 
        kwargs = dict(kwargs, 
                      l = p.parameters["w"], 
                      l_1 = p.parameters["w"] / 2)
        
        #call constructor of super
        BicycleParameters.__init__(self, **kwargs)
        
        #physical parameters
        self.bp_model = m
        self.bp_params_set = p
        self.m = p.parameters["mB"] + p.parameters["mF"] + p.parameters["mH"] + p.parameters["mR"]
        self.g = p.parameters["g"]
        
        #control parameters
        self.poles = poles
        self.gains = gains
        
        #noise / disturbance parameters
        self.p_dist_roll = p_dist_roll
        self.p_dist_steer = p_dist_steer
        self.T_dist_roll = T_dist_roll
        self.T_dist_steer = T_dist_steer
        
    def get_state_space_matrices(self, v):
        """
        Calculate the state space matrices of the Whipple-Carvallo bicycle
        for the speed v. 

        Parameters
        ----------
        v : float
            Longitudinal forward speed in m/s.

        Returns
        -------
        A, B : numpy.ndarray
            State space matrices
        """
        
        return self.bp_model.form_state_space_matrices(v=v)


class InvPendulumBicycleParameters(BicycleParameters):
    """Calculate and update the parameters of a bicycle and it's rider.

    Inherits tactical parameters from VehicleParameters and dynamic and
    social force parameters from BicycleParameters.

    Provides physical and control parameters for the inverted pendulum bicycle
    and the 2D bicycle without pendulum.

    Default pysical bike parameter values taken from:
        Moore, J. K.(2015, June 30). Bicycle Control Design in Python/v3.
        Plotly Graphing Libraries. https://plotly.com/python/v3/ipython-
        notebooks/bicycle-control-design/
    """

    def __init__(
        self,
        # rider parameters
        v_max_riding: tuple = [-1.0, 7.0],
        v_desired_default: float = 5.0,
        hfov: float = np.pi * 2 / 3,
        a_max: tuple = [-3.0, 1.0],
        a_desired_default: tuple = [-1.0, 0.5],
        # bicycle parameters
        l: float = None,
        l_1: float = 0.5,
        l_2: float = 0.5,
        delta_max: float = 1.4,
        h: float = 1.0,
        m: float = 87.0,
        i_bike_longlong: float = 3.28,
        i_steer_vertvert: float = 0.07,
        c_steer: float = 50.0,
        # control parameters
        k_p_v: float = 10.0,
        k_d0_r2: float = -600.0,
        k_d1_r2: float = 0.2,
        k_p_r1: float = 0.25,
        k_i0_r1: float = 0.2,
        # simulation parameters
        t_s: float = 0.01,
        # operational paramters
        d_arrived_inter: float = 2.0,
        d_arrived_stop: float = 2.0,
        v_max_harddecel: float = 2.5,
        v_max_stop: float = 0.6,
        v_max_walk: float = 1.5,
        delta_max_walk: tuple = 0.174,
        # repulsive force field parameters
        f_0: float = 7.0,
        e_0: float = 0.995,
        e_1: float = 0.7,
        sigma_0: float = 0.5,
        sigma_1: float = 5.0,
        sigma_2: float = 0.3,
        sigma_3: float = 4.9,
        # physical constants
        g: float = 9.81,
    ) -> None:
        """Create the parameter set of a default rider and bicycle.

        TODO:
        - Add interface to set parameters to values other then the default.
        - Add randomness.
        - Add compatibility to SUMO *.rou.xml.

        Parameters
        ----------

        v_max_riding : tuple, optional
            Maximal longitudinal velocity of the bicycle for forward and back-
            ward motion, given as [backward, forward]. The default is [-1.,7.].
        v_desired_default : float, optional
            Default desired forward velocity of the cyclist. The default is 5..
        hfov : float, optional
            Horizontal field of view of the cyclist. The vehicle will react to
            other road users within that field of view and ignore those
            outside. The default is np.pi*2/3.
        a_max : tuple, optional
            Maximum possible de-/acceleration range given as [breaking,
            accelerating]. The default is (-3.,1.).
        a_desired_default : tuple, optional
            Default desired de-/acceleration range given as [breaking,
            accelerating]. The default is (-1., .5).

        l : float, optional
            Wheelbase of the bicycle. The default is 1.0. based on
            Moore (2015).
        l_1 : float, optional
            Front section of the wheelbase of the bicycle. The default is 0.5,
            based on Moore (2015).
        l_2 : float, optional
            Front section of the wheelbase of the bicycle. The default is 0.5,
            based on Moore (2015).
        delta_max : float, optional
            Mechanical maximum of the steering angle (symmetric). The default
            is 1.4.
        h : float, optional
            Length of the inverted pendulum or hight of the center of mass
            above the ground. The default is 1.0, based on Moore (2015).
        m : float, optional
            Combined mass of the rider and the bicycle, based on Moore (2015).
        i_bike_longlong : float, optional
            Rotational moment of inertia of the bicycle around the longitudinal
            axis on the ground. The default is 3.28, based on Moore (2015).
        i_steer_vertvert : float, optional
            Rotational moment of interia of the steer column around it's
            vertical axis of rotation. The default is 0.07.
        c_steer : float, optional
            Damping coefficent of the steer column dynamics. The default is 50.

        k_p_v : float, optional
            Proportional gain for velocity control. The default is 10.0.
        k_d0_r2 : float, optional
            Linear factor of the differential gain of controler R2 for steer/
            lean angle control. The default is -600. Must be negative.
        k_d1_r2 : float, optional
            Speed offset for the differential gain of controler R2
            for steer/lean angle control. The default is 0.2.
        k_p_r1 : float, optional
            Constant proportional gain of the yaw angle
            controler R1. The default is 0.25.
        k_i0_r1 : float, optional
            Linear factor of the speed-adaptive integral gain of the yaw angle
            controler R1. The default is 0.2.

        t_s : float, optional
            Sample time. The default is 0.01.

        d_arrived_inter : float, optional
            Distance to intermediate destination at which the destination
            counts are "arrived" and the vehicle continues to the next
            destination. The default is 2.0.
        d_arrived_stop : float, optional
            Distance to stop destination at which the destination counts as
            "arrived" and the vehicle may stop. The default is 2.0.
        v_max_stop : float, optional
            Maximum speed below which the vehcile may count as stopped.
            The default is 0.1.
        v_max_harddecel : float, optional
            Maximum speed below which the vehicle may apply hard decceleration
            for a normal stop at a stop destination. The default is 2.5.
        v_max_walk : float, optional
            Maximum speed above which the rider stops walking and starts
            riding. The default is 1.5.
        delta_max_walk : float, optional
            Maximum steer angle (symmetric) below which the rider may stop
            walking and start cycling. The default is 0.174.

        f_0 : float, optional
            Repulsive force magnitude at the ego-location of the bike. The
            default is 7.0 (m/s)
        e_0 : float, optional
            Exccentricity of the repulsive force ellipses, The default
            is 0.995.
        e_1 : float, optional
            Relative orientation modulation factor for the excentricity of
            the repulsive force ellipses. The default is 0.7.
        sigma_0 : float, optional
            Radial decay of the repulsive force. The default is 0.5.
        sigma_1 : float, optional
            Relative orientation modulation factor for the radial decay.
            The default is 5.0.
        sigma_2 : float, optional
            Relative radial position modulation factor for the radial decay.
            The default is 0.3.
        sigma_3 : float, optional
            Relative radial position and relative orientation cross-modulation
            factor for the radial decay. The default is 4.9.

        g : float, optional
            Graviational constant.


        Returns
        -------
        None
            DESCRIPTION.
        """

        BicycleParameters.__init__(
            self,
            v_max_riding=v_max_riding,
            v_desired_default=v_desired_default,
            hfov=hfov,
            a_max=a_max,
            a_desired_default=a_desired_default,
            l=l,
            l_1=l_1,
            l_2=l_2,
            delta_max=delta_max,
            k_p_v=k_p_v,
            t_s=t_s,
            d_arrived_inter=d_arrived_inter,
            d_arrived_stop=d_arrived_stop,
            v_max_stop=v_max_stop,
            v_max_harddecel=v_max_harddecel,
            f_0=f_0,
            e_0=e_0,
            e_1=e_1,
            sigma_0=sigma_0,
            sigma_1=sigma_1,
            sigma_2=sigma_2,
            sigma_3=sigma_3,
        )

        # Bike dimensions
        self.h = h
        self.m = m
        self.i_bike_longlong = i_bike_longlong
        self.i_steer_vertvert = i_steer_vertvert
        self.c_steer = c_steer

        # Control
        self.k_d0_r2 = k_d0_r2
        self.k_d1_r2 = k_d1_r2
        self.k_p_r1 = k_p_r1
        self.k_i0_r1 = k_i0_r1

        #
        self.v_max_walk = v_max_walk
        self.delta_max_walk = delta_max_walk

        # Pysical constants
        self.g = g

        # combined parameters
        self.tau_1_squared = (self.i_bike_longlong + self.m * self.h**2) / (
            self.m * self.g * self.h
        )

    # ---- PROPERTIES ----

    @property
    def m(self) -> float:
        return self._m

    @m.setter
    def m(self, m) -> None:
        if hasattr(self, "_m"):
            raise AttributeError("m is immutable.")
        if not isinstance(m, float):
            raise TypeError("m must be a float.")
        if not m >= 0:
            raise ValueError(
                f"m must be >=0, instead it was \
                             {m:.2f}"
            )
        self._m = m

    @property
    def h(self) -> float:
        return self._h

    @h.setter
    def h(self, h) -> None:
        if hasattr(self, "_h"):
            raise AttributeError("h is immutable.")
        if not isinstance(h, float):
            raise TypeError("h must be a float.")
        if not h >= 0:
            raise ValueError(
                f"h must be >=0, instead it was \
                             {h:.2f}"
            )
        self._h = h

    @property
    def i_bike_longlong(self) -> float:
        return self._i_bike_longlong

    @i_bike_longlong.setter
    def i_bike_longlong(self, i_bike_longlong) -> None:
        if hasattr(self, "_i_bike_longlong"):
            raise AttributeError("i_bike_longlong is immutable.")
        if not isinstance(i_bike_longlong, float):
            raise TypeError("i_bike_longlong must be a float.")
        if not i_bike_longlong >= 0:
            raise ValueError(
                f"i_bike_longlong must be >=0, instead it was \
                             {i_bike_longlong:.2f}"
            )
        self._i_bike_longlong = i_bike_longlong

    @property
    def i_steer_vertvert(self) -> float:
        return self._i_steer_vertvert

    @i_steer_vertvert.setter
    def i_steer_vertvert(self, i_steer_vertvert) -> None:
        if hasattr(self, "_i_steer_vertvert"):
            raise AttributeError("i_steer_vertvert is immutable.")
        if not isinstance(i_steer_vertvert, float):
            raise TypeError("i_steer_vertvert must be a float.")
        if not i_steer_vertvert >= 0:
            raise ValueError(
                f"i_steer_vertvert must be >=0, instead it was \
                             {i_steer_vertvert:.2f}"
            )
        self._i_steer_vertvert = i_steer_vertvert

    @property
    def c_steer(self) -> float:
        return self._c_steer

    @c_steer.setter
    def c_steer(self, c_steer) -> None:
        if hasattr(self, "_c_steer"):
            raise AttributeError("c_steer is immutable.")
        if not isinstance(c_steer, float):
            raise TypeError("c_steer must be a float.")
        if not c_steer >= 0:
            raise ValueError(
                f"c_steer must be >=0, instead it was \
                             {c_steer:.2f}"
            )
        self._c_steer = c_steer

    @property
    def k_d0_r2(self) -> float:
        return self._k_d0_r2

    @k_d0_r2.setter
    def k_d0_r2(self, k_d0_r2) -> None:
        if hasattr(self, "_k_d0_r2"):
            raise AttributeError("k_d0_r2 is immutable.")
        if not isinstance(k_d0_r2, float):
            raise TypeError("k_d0_r2 must be a float.")
        if not k_d0_r2 < 0:
            raise ValueError(
                f"k_d0_r2 must be <0 to stabilize the lean/steer \
                             angle loop, instead it was \
                             {k_d0_r2:.2f}"
            )
        self._k_d0_r2 = k_d0_r2

    @property
    def k_d1_r2(self) -> float:
        return self._k_d1_r2

    @k_d1_r2.setter
    def k_d1_r2(self, k_d1_r2) -> None:
        if hasattr(self, "_k_d1_r2"):
            raise AttributeError("k_d1_r2 is immutable.")
        if not isinstance(k_d1_r2, float):
            raise TypeError("k_d1_r2 must be a float.")
        self._k_d1_r2 = k_d1_r2

    @property
    def k_p_r1(self) -> float:
        return self._k_p_r1

    @k_p_r1.setter
    def k_p_r1(self, k_p_r1) -> None:
        if hasattr(self, "_k_p_r1"):
            raise AttributeError("k_p_r1 is immutable.")
        if not isinstance(k_p_r1, float):
            raise TypeError("k_p_r1 must be a float.")
        if not k_p_r1 >= 0:
            raise ValueError(
                f"k_p_r1 must be >=0, instead it was \
                             {k_p_r1:.2f}"
            )
        self._k_p_r1 = k_p_r1

    @property
    def k_i0_r1(self) -> float:
        return self._k_i0_r1

    @k_i0_r1.setter
    def k_i0_r1(self, k_i0_r1) -> None:
        if hasattr(self, "_k_i0_r1"):
            raise AttributeError("k_i0_r1 is immutable.")
        if not isinstance(k_i0_r1, float):
            raise TypeError("k_i0_r1 must be a float.")
        if not k_i0_r1 >= 0:
            raise ValueError(
                f"k_i0_r1 must be >=0, instead it was \
                             {k_i0_r1:.2f}"
            )
        self._k_i0_r1 = k_i0_r1

    @property
    def v_max_walk(self) -> float:
        return self._v_max_walk

    @v_max_walk.setter
    def v_max_walk(self, v_max_walk) -> None:
        if hasattr(self, "_v_max_walk"):
            raise AttributeError("v_max_walk is immutable.")
        if not isinstance(v_max_walk, float):
            raise TypeError("v_max_walk must be a float.")
        if not v_max_walk >= 0:
            raise ValueError(
                f"v_max_walk must be >=0, instead it was \
                             {v_max_walk:.2f}"
            )
        self._v_max_walk = v_max_walk

    @property
    def delta_max_walk(self) -> float:
        return self._delta_max_walk

    @delta_max_walk.setter
    def delta_max_walk(self, delta_max_walk) -> None:
        if hasattr(self, "_delta_max_walk"):
            raise AttributeError("delta_max_walk is immutable.")
        if not isinstance(delta_max_walk, float):
            raise TypeError("delta_max_walk must be a float.")
        if not 0 < delta_max_walk <= np.pi:
            raise ValueError(
                f"delta_max_walk must be in ]0,pi], instead it was \
                             {delta_max_walk:.2f}"
            )
        self._delta_max_walk = delta_max_walk

    # ---- METHODS ----

    def timevarying_combined_params(self, v: float) -> tuple[float, float]:
        """Calculate the time-varying (speed-dependend) combined parameters
        tau_2 and K of the lean angle dynamics (G_theta).

        Parameters
        ----------
        v : float
            Current speed of the bicycle.

        Returns
        -------
        K : float
            Parameter K.
        K_tau_2 : float
            Parameter K * tau_2

        """

        K_tau_2 = (v * self.l_2) / (self.g * (self.l))
        K = (v**2) / (self.g * (self.l))

        tau_3 = self.l / v

        return K, K_tau_2, tau_3

    def fullstate_feedback_gains(self, v):
        # K_x = np.array(
        #    [[6.26092881, -48.635, -6.92845026, -2.25215286, -2.15918001]]
        # )
        # K_u = -2.1591800063357907

        params_kx = np.array(
            [
                [3.48203226e02, -5.12057324e03, 1.58364873e04, -1.98073306e04],
                [-4.51700000e01, 0.00000000e00, 0.00000000e00, 0.00000000e00],
                [-9.16379250e02, 1.31769807e04, -6.57341643e04, 8.22163589e04],
                [3.20214069e02, -4.69953797e03, 1.66378680e04, -2.43114309e04],
                [2.87549256e-08, -2.27913445e03, 0.00000000e00, 0.00000000e00],
            ]
            #[
            # [ 1.27977414e+02, -1.94670000e+03,  2.43962111e+03, -3.02454886e+03],
            # [-4.51700000e+01,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
            # [-2.26494120e+00,  2.85310350e+00, -1.01263905e+04,  1.25543113e+04],
            # [ 9.05946737e+00, -1.99832338e+02, -2.45608874e+03, -2.08063358e+02],
            # [-3.38428663e-09, -2.27913445e+03,  0.00000000e+00,  0.00000000e+00],
            # ]
        )

        params_ku = np.array(
            #[2.87524813e-08, -2.27913445e03, 0.00000000e00, 0.00000000e00]
            [-3.38638984e-09, -2.27913445e+03,  0.00000000e+00,  0.00000000e+00]
        )

        vdata = np.array((1, v**-1, v**-2, v**-3))

        K_x = params_kx @ vdata
        K_u = params_ku @ vdata

        K_x = K_x[np.newaxis, :]

        return K_x, K_u

    def update_dynamic_params(
        self, v: float
    ) -> tuple[tuple, tuple, tuple, tuple]:
        """Calculate the speed-dependend parameters of the model dynamics in
        z-domain.

        Uses the control toolbox to convert the discretize the time-continous
        systems.

        TODO: Check if this is slow.

        Parameters
        ----------
        v : float
            Current speed of the bicycle.

        Returns
        -------
        params_r1 : tuple[float, float]
            Parameters of the controler R1 dynamics given as (a,b).
        params_r2_delta : tuple[float, float]
            Parameters of the R2 * G_delta dynamics given as (a,b).
        params_theta : tuple[float, float]
            Parameters of the G_theta dynamics given as (a,b).
        params_psi : tuple[float, float]
            Parameters of the G_psi dynamics given as (a,b).
        """

        # time-varying parameters
        K, K_tau_2 = self.timevarying_combined_params(v)

        # controller gains [Kp, Ki, Kd]
        K_r2 = self.r2_adaptive_gain(v)
        K_r1 = self.r1_adaptive_gain(v)

        # Transfer function steer->yaw
        G_psi = ct.tf((1), ((self.l) / v, 0))

        # Transfer function for steer->lean
        G_theta = ct.tf((-K_tau_2, -K), (self.tau_1_squared, 0, -1))

        # Transfer functions for steering with inertia
        G_delta = ct.tf((1), (self.i_steer_vertvert, self.c_steer, 0))

        # Transfer functions of controllers
        G_r2 = ct.tf((K_r2[2], 0), (1))
        G_r1 = ct.tf((K_r1[2], K_r1[0], K_r1[1]), (1, 0))

        # Time-discrete transfer functions
        Gz_r1 = ct.sample_system(G_r1, self.t_s)
        Gz_psi = ct.sample_system(G_psi, self.t_s)
        Gz_r2_delta = ct.sample_system(ct.series(G_r2, G_delta), self.t_s)
        Gz_theta = ct.sample_system(G_theta, self.t_s)

        return (
            (Gz_r1.den[0][0], Gz_r1.num[0][0]),
            (Gz_r2_delta.den[0][0], Gz_r2_delta.num[0][0]),
            (Gz_theta.den[0][0], Gz_theta.num[0][0]),
            (Gz_psi.den[0][0], Gz_psi.num[0][0]),
        )

    def min_stable_speed_inner(self) -> float:
        """Calculate the speed v_min below which the inner loop becomes
        unstable.

        Returns
        -------
        v_min : float
            Minimum stable speed.
        """

        x = self.k_d0_r2
        y = self.c_steer * self.g * (self.l_1 + self.l_2)
        z = self.c_steer * self.g * (self.l_1 + self.l_2) * self.k_d1_r2
        v_min = (-y - np.sqrt(y**2 - 4 * x * z)) / (2 * x)

        return v_min