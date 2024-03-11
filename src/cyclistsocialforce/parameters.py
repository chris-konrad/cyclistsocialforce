# -*- coding: utf-8 -*-
"""
Created on Tue Sep 26 12:16:53 2023.

Classes managing vehicle parameter sets. 

@author: Christoph M. Schmidt
"""

import numpy as np
import control as ct


class RoadElementParameters:
    def __init__(
        self,
        roadsurface_color=(0.8, 0.8, 0.8),
        roadedge_color="white",
        roadedge_linewidth=1,
    ):
        self.roadsurface_color = roadsurface_color

        self.roadedge_color = roadedge_color
        self.roadedge_linewidth = roadedge_linewidth


class VehicleParameters:
    """Calculate and update the parameters of a vehicle.

    Provides tactical parameters.
    """

    def __init__(
        self,
        t_s: float = 0.01,
        d_arrived_inter: float = 2.0,
        d_arrived_stop: float = 2.0,
        v_max_stop: float = 0.1,
        v_max_harddecel: float = 2.5,
        hfov: float = 2 * np.pi,
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
        self.t_s = t_s
        self.d_arrived_inter = d_arrived_inter
        self.d_arrived_stop = d_arrived_stop
        self.v_max_stop = v_max_stop
        self.v_max_harddecel = v_max_harddecel
        self.hfov = hfov

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
        v_max_riding: tuple = [-1.0, 7.0],
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
        sigma_0: float = 3.0,
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

        # repulsive force fields
        self.f_0 = f_0
        self.e_0 = e_0
        self.e_1 = e_1
        self.sigma_0 = sigma_0
        self.sigma_1 = sigma_1
        self.sigma_2 = sigma_2
        self.sigma_3 = sigma_3

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

    @property
    def f_0(self) -> float:
        return self._f_0

    @v_max_walk.setter
    def f_0(self, f_0) -> None:
        if hasattr(self, "_f_0"):
            raise AttributeError("f_0 is immutable.")
        if not isinstance(f_0, float):
            raise TypeError("f_0 must be a float.")
        if not f_0 >= 0:
            raise ValueError(
                f"f_0 must be >=0, instead it was \
                             {f_0:.2f}"
            )
        self._f_0 = f_0

    @property
    def e_0(self) -> float:
        return self._e_0

    @e_0.setter
    def e_0(self, e_0) -> None:
        if hasattr(self, "_e_0"):
            raise AttributeError("e_0 is immutable.")
        if not isinstance(e_0, float):
            raise TypeError("e_0 must be a float.")
        if not 0 < e_0 < 1:
            raise ValueError(
                f"e_0 must be in ]0,1[, instead it was \
                             {e_0:.2f}"
            )
        self._e_0 = e_0

    @property
    def e_1(self) -> float:
        return self._e_1

    @e_1.setter
    def e_1(self, e_1) -> None:
        if hasattr(self, "_e_1"):
            raise AttributeError("e_1 is immutable.")
        if not isinstance(e_1, float):
            raise TypeError("e_1 must be a float.")
        if not 0 < e_1 < 1:
            raise ValueError(
                f"e_1 must be in ]0,1[, instead it was \
                             {e_1:.2f}"
            )
        self._e_1 = e_1

    @property
    def sigma_0(self) -> float:
        return self._sigma_0

    @sigma_0.setter
    def sigma_0(self, sigma_0) -> None:
        if hasattr(self, "_sigma_0"):
            raise AttributeError("sigma_0 is immutable.")
        if not isinstance(sigma_0, float):
            raise TypeError("sigma_0 must be a float.")
        if not sigma_0 >= 0:
            raise ValueError(
                f"sigma_0 must be >=0, instead it was \
                             {sigma_0:.2f}"
            )
        self._sigma_0 = sigma_0

    @property
    def sigma_1(self) -> float:
        return self._sigma_1

    @sigma_1.setter
    def sigma_1(self, sigma_1) -> None:
        if hasattr(self, "_sigma_1"):
            raise AttributeError("sigma_1 is immutable.")
        if not isinstance(sigma_1, float):
            raise TypeError("sigma_1 must be a float.")
        if not sigma_1 >= 0:
            raise ValueError(
                f"sigma_1 must be >=0, instead it was \
                             {sigma_1:.2f}"
            )
        self._sigma_1 = sigma_1

    @property
    def sigma_2(self) -> float:
        return self._sigma_2

    @sigma_2.setter
    def sigma_2(self, sigma_2) -> None:
        if hasattr(self, "_sigma_2"):
            raise AttributeError("sigma_2 is immutable.")
        if not isinstance(sigma_2, float):
            raise TypeError("sigma_2 must be a float.")
        if not sigma_2 >= 0:
            raise ValueError(
                f"sigma_2 must be >=0, instead it was \
                             {sigma_2:.2f}"
            )
        self._sigma_2 = sigma_2

    @property
    def sigma_3(self) -> float:
        return self._sigma_3

    @sigma_3.setter
    def sigma_3(self, sigma_3) -> None:
        if hasattr(self, "_sigma_3"):
            raise AttributeError("sigma_3 is immutable.")
        if not isinstance(sigma_3, float):
            raise TypeError("sigma_3 must be a float.")
        if not sigma_3 >= 0:
            raise ValueError(
                f"sigma_3 must be >=0, instead it was \
                             {sigma_3:.2f}"
            )
        self._sigma_3 = sigma_3

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

    def fullstate_feedback_gains(self):
        K_x = np.array(
            [[6.26092881, -48.635, -6.92845026, -2.25215286, -2.15918001]]
        )
        K_u = -2.1591800063357907

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
