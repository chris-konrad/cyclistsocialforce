# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 11:55:09 2024

@author: Christoph M. Konrad
"""

import numpy as np
import control as ct

from cyclistsocialforce.utils import (
    limitAngle,
    thresh,
    cart2polar,
    angleDifference,
)

class PIDcontroller:
    
    def __init__(self, kp, ki, kd, dT, isangle=False):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.dT = dT
        self.e = 0
        self.i = 0
        self.isangle=isangle
        self.hist = []
        
    def step(self, e):
        
        #differential component
        if 0:
            de = angleDifference(self.e, e)
        else:
            de = self.e - e
        self.d = self.kd * de/self.dT
        self.e = e
        
        #integral component
        self.i = self.i + (self.ki * self.e * self.dT)    
        
        #proportional component
        self.p = self.kp * self.e
        
        # output
        out = self.p + self.i + self.d
        
        self.hist.append(e)
        
        return out

class Dynamics:

    def __init__(self, Vehicle):
        self.x = Vehicle.s

    def step(self, Vehicle, F1, F2):
        pass


class PPointSpeedDynamics(Dynamics):
    """A class describing the p-controlled point model of the form:

    dv(t)/dt - Kp (v(t) - v_ref) = 0,

    which derives the acceleration of a point proportionally from the
    difference between the current speed and a desired speed v_ref.

    Assumes constant simulation time t_s and constant gain K_p.
    """

    def __init__(self, vehicle):

        # precalc the constant exponent of the solution
        self.exponent = np.exp(-vehicle.params.k_p_v * vehicle.params.t_s)

    def step(self, vehicle, v_ref):
        """Advance the speed dynamics by one time step.

        This directly implements the solution of the differential equation
        describing the speed dynamics.

        Parameters
        ----------
        vehicle : cyclistsocialforce.vehicle.Vehicle
            The Vehicle with this SpeedDynamics object.
        v_d : float
            The reference speed.
        """

        vehicle.s[3] = v_ref + (vehicle.s[3] - v_ref) * self.exponent


class PlanarTwoWheelerDynamics(Dynamics):
    """A class describing the dynamics of a planar bicycle.

    States: x = (delta, ddelta, psi)

    with steer angle delta, steer angle rate ddelta, and yaw angle psi.
    """

    def __init__(self, bicycle):

        # poles
        self.poles = bicycle.params.poles

        # geometry
        self.w = bicycle.params.l  # wheelbase

        # save initial state
        self.x = np.zeros(2)
        self.x[0] = bicycle.s[4]
        self.x[1] = bicycle.s[2]

        # yaw dynamics
        self.update(bicycle.s[3])

        # speed dynamics
        self.speed_dynamics = PPointSpeedDynamics(bicycle)

    def update(self, v):
        """
        Update the state matrix of the planar bicycle dynamics for the
        current speed.

        Parameters
        ----------
        v : float
            Current forward speed.
        """

        A = np.zeros((2, 2))
        B = np.array(((1,), (0,)))
        C = np.array((0, 1))
        D = np.zeros((1, 1))

        A[1, 0] = v / self.w

        self.sys, self.gains = from_pole_placement(A, B, C, D, self.poles)

    def step(self, bicycle, Fx, Fy):

        # update statespace parameters with current speed
        self.update(bicycle.s[3])

        # absolute force angle
        psi_d = np.arctan2(Fy, Fx)
        v_d = np.sqrt(Fy**2 + Fx**2)

        # calculate steer angle for stabilization
        results = ct.forced_response(
            self.sys,
            T=np.array([0, bicycle.params.t_s]),
            X0=self.x,
            return_x=True,
            U=np.ones(2) * psi_d,
            squeeze=False,
        )
        self.x = results.states[:, 1]

        bicycle.s[2] = limitAngle(results.states[1, 1])
        bicycle.s[4] = limitAngle(results.states[0, 1])

        self.speed_dynamics.step(bicycle, v_d)

        y = bicycle.s[1] + bicycle.params.t_s * bicycle.s[3] * np.sin(
            bicycle.s[2]
        )
        x = bicycle.s[0] + bicycle.params.t_s * bicycle.s[3] * np.cos(
            bicycle.s[2]
        )

        bicycle.s[0] = x
        bicycle.s[1] = y


class WhippleCarvalloDynamics(Dynamics):
    """ Implementing the dynamics of a Bicycle using the linearized
    Whipple-Carvallo model (Meijaard et al., 2027) and Jason Moore's bicycle
    parameters toolbox (https://bicycleparameters.readthedocs.io/en/latest/)
    together with full-state feedback control. 
    
    Literature
    ----------
    
    Meijaard, J. p, Papadopoulos, J. M., Ruina, A., & Schwab, A. l. (2007). 
        Linearized dynamics equations for the balance and steer of a bicycle: 
        A benchmark and review. Proceedings of the Royal Society A: 
        Mathematical, Physical and Engineering Sciences, 463(2084), 1955–1982. 
        https://doi.org/10.1098/rspa.2007.1857
    """

    def __init__(self, bicycle):

        self.bp_model = bicycle.params.bp_model
        
        # init dynamics either from poles or from gains provided in the params
        # object
        try:
            self.from_gains = not(bicycle.params.gains is None)
        except AttributeError:
            self.from_gains = False    
            assert not(bicycle.params.poles is None), (f"The parameters object"
                             f" has to have either a poles or a gains"
                             f"  property!")
            
        if self.from_gains:
            self.desired_gains = bicycle.params.gains
            self.desired_poles = None
        else:
            self.desired_gains = None
            self.desired_poles = bicycle.params.poles
            
        # get geometry parameters
        w = self.bp_model.parameter_set.parameters["w"]
        c = self.bp_model.parameter_set.parameters["c"]
        coslam = np.cos(self.bp_model.parameter_set.parameters["lam"])

        # pre-calc yaw state matrix coefficients
        self.A41_over_v = coslam / w
        self.A43 = coslam * c / w

        # initialize state space system
        self.update(bicycle.s[3])

        # save initial state x0 = (phi, delta, dphi, ddelta, psi)
        self.x = np.zeros(5)
        self.x[4] = bicycle.s[2]
        self.x[0] = bicycle.s[5]
        self.x[1] = bicycle.s[4]

        # speed controller
        self.speed_controller = PIDcontroller(
            bicycle.params.k_p_v, 0, 0, bicycle.params.t_s, isangle=False
        )

        self.psi_boundless = bicycle.s[2]

        # roll and steer torque disturbance
        self.p_dist_roll = bicycle.params.p_dist_roll
        self.p_dist_steer = bicycle.params.p_dist_steer
        self.T_dist_roll = bicycle.params.T_dist_roll
        self.T_dist_steer = bicycle.params.T_dist_steer
        
        # names
        self.state_names = ['phi', 'delta', 'phidot', 'deltadot', 'psi']
        self.input_names = ['T_phi', 'T_delta']
        self.param_names = ['k_'+s for s in self.state_names]

    def get_statespace_matrices(self, v):
        Awc, Bwc = self.bp_model.form_state_space_matrices(v=v)

        # add yaw dynamics
        A = np.zeros((5, 5))
        A[:4, :4] = Awc
        A[4, 1] = self.A41_over_v * v
        A[4, 3] = self.A43

        B = np.zeros((5, 2))
        B[:4, :] = Bwc

        # output
        C = np.zeros((1, A.shape[1]))
        C[0, 4] = 1
        D = np.zeros((C.shape[0], B.shape[1]))

        return A, B, C, D

    def update(self, v):
        A, B, C, D = self.get_statespace_matrices(v)

        # recreate system
        if self.from_gains:
            self.sys, self.gains = from_gains(
                A, B[:, 1][:, np.newaxis], C, D[:, 1][:, np.newaxis], 
                self.desired_gains)
        else:  
            self.sys, self.gains = from_pole_placement(
                A, B[:, 1][:, np.newaxis], C, D[:, 1][:, np.newaxis],
                self.desired_poles)

        # disturbance inputs
        self.add_disturbance_inputs(B)

    def step(self, bicycle, Fx, Fy):

        n_turns = int(self.psi_boundless / (2 * np.pi))

        # update statespace parameters with current speed
        self.update(bicycle.s[3])

        # absolute force angle
        psi_d = np.arctan2(Fy, Fx)
        psi_d = psi_d + n_turns * 2 * np.pi

        psi_d_temp = np.array(
            [psi_d - (2 * np.pi), psi_d, psi_d + (2 * np.pi)]
        )
        i_temp = np.argmin(np.abs(psi_d_temp - self.psi_boundless))
        psi_d = psi_d_temp[i_temp]

        # if abs(psi_d - bicycle.s[2]) > np.pi:
        #    dpsi = angleDifference(psi_d, bicycle.s[2])
        #    psi_d = bicycle.s[2] + dpsi

        rng = np.random.default_rng()
        z_phi = (
            self.T_dist_roll
            * rng.binomial(1, self.p_dist_roll)
            * (1 - 2 * rng.binomial(1, 0.5))
        )
        z_delta = (
            self.T_dist_steer
            * rng.binomial(1, self.p_dist_steer)
            * (1 - 2 * rng.binomial(1, 0.5))
        )

        if z_phi != 0.0:
            print(f"{z_phi:.1f} N roll torque disturbance!")

        if z_delta != 0.0:
            print(f"{z_delta:.1f} N steer torque disturbance!")

        u = np.array([psi_d, z_phi, z_delta])
        U = np.c_[u, u]

        # calculate steer angle for stabilization
        results = ct.forced_response(
            self.sys,
            T=np.array([0, bicycle.params.t_s]),
            X0=self.x,
            return_x=True,
            U=U,
            squeeze=False,
        )
        self.x = results.states[:, 1]  # (roll, steer, droll, dsteer, yaw)

        self.psi_boundless = results.states[4, 1]
        bicycle.s[2] = limitAngle(results.states[4, 1])  # yaw
        bicycle.s[4] = limitAngle(results.states[1, 1])  # steer
        bicycle.s[5] = limitAngle(results.states[0, 1])  # roll

        if bicycle.saveForces:
            bicycle.trajF[0, bicycle.i] = psi_d

        self.step_pos(bicycle, Fx, Fy)

    def speed_control(self, v, vd):
        """Calculate the acceleration as a reaction to the current social
        force.

        Parameters
        ----------

        vd : float
            Desired speed.

        Returns
        -------
        a : float
            Acceleration

        """
        dv = vd - v
        a = self.speed_controller.step(dv)

        return a

    def step_pos(self, bicycle, Fx, Fy):
        """Propagate the speed dynamics by one time step and integrate
        to calculate the position.

        Parameters
        ----------
        Fx : float
            X-component of the current social force.
        Fy : float
            y-component of the current social force.

        Returns
        -------
        s : list of floats
            Next position and speed of the
            bicycle given as s = (x, y, v)

        """

        vd = np.sqrt(Fx**2 + Fy**2)

        a = self.speed_control(bicycle.s[3], vd)

        a = thresh(a, bicycle.params.a_max)
        v = bicycle.s[3] + bicycle.params.t_s * a
        v = thresh(v, bicycle.params.v_max_riding)
        # print(v)
        y = bicycle.s[1] + bicycle.params.t_s * v * np.sin(bicycle.s[2])
        x = bicycle.s[0] + bicycle.params.t_s * v * np.cos(bicycle.s[2])

        bicycle.s[0] = x
        bicycle.s[1] = y
        bicycle.s[3] = v

    def add_disturbance_inputs(self, Bb):
        """
        Add roll and/or steer torque disturbance inputs to the statespace
        system.

        Parameters
        ----------
        Bb : array-like
            Input matrix of the the linear Whipple-Carvallo bicycle model

        """

        # if not (self.add_dist_steer or self.add_dist_roll):
        #    pass

        BdKu = self.sys.B  # bike yaw error -> steer torqe input
        B = BdKu

        # if self.add_dist_roll:
        B = np.c_[B, Bb[:, 0]]  # bike roll torque disturbance input

        # if self.add_dist_steer:
        B = np.c_[B, Bb[:, 1]]  # bike steer torque disturbance input
        D = np.zeros((self.sys.C.shape[0], B.shape[1]))

        self.sys = ct.StateSpace(self.sys.A, B, self.sys.C, D)


class HessBikeRiderDynamics(WhippleCarvalloDynamics):
    """ Implementing the dynamics of a Bicycle using the linearized
    Whipple-Carvallo model (Meijaard et al., 2027) and Jason Moore's bicycle
    parameters toolbox (https://bicycleparameters.readthedocs.io/en/latest/)
    together with the human control model proposed by Hess et al. (2012).
    
    Literature
    ----------
    
    Hess, R., Moore, J. K., & Hubbard, M. (2012). Modeling the Manually 
        Controlled Bicycle. IEEE Transactions on Systems, Man, and 
        Cybernetics - Part A: Systems and Humans, 42(3), 545–557. IEEE 
        Transactions on Systems, Man, and Cybernetics - Part A: Systems and 
        Humans. https://doi.org/10.1109/TSMCA.2011.2164244

    Meijaard, J. p, Papadopoulos, J. M., Ruina, A., & Schwab, A. l. (2007). 
        Linearized dynamics equations for the balance and steer of a bicycle: 
        A benchmark and review. Proceedings of the Royal Society A: 
        Mathematical, Physical and Engineering Sciences, 463(2084), 1955–1982. 
        https://doi.org/10.1098/rspa.2007.1857
    """

    def __init__(self, Bicycle):
        WhippleCarvalloDynamics.__init__(self, Bicycle)

        # save initial state x0 = (psi, delta, phi, ddelta, dphi, Tdelta, dTdelta)
        self.x = np.r_[self.x, np.zeros(2)]

    def get_adaptive_gains(self, v):
        """
        Gain curves eyeballed from Moore (2012)
        """
        k_delta = 43  # 58/10 * v
        k_dphi = -0.08  # 3.3/10 * v - 3
        k_phi = 8.5  # 3.3
        k_psi = 0.173  # 2.25/10 * v - 0.5
        omega = 28
        zeta = np.sqrt(2) / 2

        return k_delta, k_phi, k_dphi, k_psi, omega, zeta

    def get_statespace_matrices(self, v):
        """
        Construct the statespace matrices

        Parameters
        ----------
        v : float
            Current forward speed.

        """
        # Get adaptive gains
        k_delta, k_phi, k_dphi, k_psi, omega, zeta = self.get_adaptive_gains(v)

        # build statespace matrices
        A = np.zeros((7, 7))
        A[5, 6] = 1
        A[6, :] = np.array(
            [
                -k_delta * k_phi * k_dphi * omega**2,
                -k_delta * omega**2,
                -k_delta * k_dphi * omega**2,
                0,
                -k_delta * k_phi * k_dphi * k_psi * omega**2,
                -(omega**2),
                -2 * omega * zeta,
            ]
        )

        B = np.zeros((7, 1))
        B[6] = k_delta * k_phi * k_dphi * k_psi * omega**2

        # output
        C = np.zeros((1, A.shape[1]))
        C[0, 4] = 1
        D = np.zeros((C.shape[0], B.shape[1]))

        # Add the open-loop whipple model
        Awc, Bwc, Cwc, Dwc = WhippleCarvalloDynamics.get_statespace_matrices(
            self, v
        )

        Bwc = Bwc[:, 1]  # discard roll torque input

        A[0:5, 0:5] = Awc
        A[0:5, 5] = Bwc.flatten()

        return A, B, C, D

    def update(self, v):
        A, B, C, D = self.get_statespace_matrices(v)
        self.sys = ct.StateSpace(A, B, C, D)


class ParticleDynamicsXY(Dynamics):
    """ A class for modelling the dynamics of a bicycle as a mass-less 
    particle in planar space"""

    def __init__(self, Vehicle):

        self.poles = Vehicle.params.poles

        # init state space system
        A = np.array([[0, 0, 1, 0], [0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0]])
        B = np.array(([[0, 0], [0, 0], [1, 0], [0, 1]]))
        C = np.array([[0, 0, 1, 0], [0, 0, 0, 1]])
        D = np.zeros((C.shape[0], B.shape[1]))

        self.sys, self.gains = from_pole_placement(A, B, C, D, self.poles)

        # save initial state x = [px, py, vx, vy]
        self.x = np.zeros(4)
        self.x[0:2] = Vehicle.s[0:2]
        self.x[2] = Vehicle.s[3] * np.cos(Vehicle.s[2])
        self.x[3] = Vehicle.s[3] * np.sin(Vehicle.s[2])

    def step(self, Vehicle, Fx, Fy):
        """Advance the dynamic model by one time step and update the state
        of Vehicle accordingly
        """

        results = ct.forced_response(
            self.sys,
            T=np.array([0, Vehicle.params.t_s]),
            X0=self.x,
            U=(np.ones(2) * np.array(((Fx,), (Fy,)))),
            return_x=True,
            squeeze=False,
        )

        self.x = results.states[:, 1]

        temp, psi_i = cart2polar(
            (results.states[2, 1]), (results.states[3, 1])
        )

        Vehicle.s[0:2] = results.states[0:2, 1]
        Vehicle.s[2] = psi_i
        Vehicle.s[3] = np.sqrt(np.sum(results.states[2:4, 1] ** 2))
        
class ParticleDynamicsHelbingMolnar(ParticleDynamicsXY):
    """ A class for modelling the dynamics of a bicycle as a mass-less particle 
    in planar space with simple heading + speed tracking as done in Helbing & 
    Molnar's (1995) original social force model.
    
    Literature
    ----------
    Helbing, D., & Molnár, P. (1995). Social force model for pedestrian 
        dynamics. Physical Review E, 51(5), 4282–4286. 
        https://doi.org/10.1103/PhysRevE.51.4282
    """
    
    def __init__(self, Vehicle):
        
        tau = 0.1
        self.gains = [1/tau]
        
        # init state space system
        A = np.array([[0, 0, 1, 0], [0, 0, 0, 1], [0, 0, -1/tau, 0], [0, 0, 0, -1/tau]])
        B = np.array(([[0, 0], [0, 0], [1/tau, 0], [0, 1/tau]]))
        C = np.array([[0, 0, 1, 0], [0, 0, 0, 1]])
        D = np.zeros((C.shape[0], B.shape[1]))
    
        self.sys = ct.StateSpace(A, B, C, D)
        
        # save initial state x = [px, py, vx, vy]
        self.x = np.zeros(4)
        self.x[0:2] = Vehicle.s[0:2]
        self.x[2] = Vehicle.s[3] * np.cos(Vehicle.s[2])
        self.x[3] = Vehicle.s[3] * np.sin(Vehicle.s[2])
        
        
def test_stability(sys, stability_type = "asymptotical"):
    """
    Test if a dynamic system is stable.

    Parameters
    ----------
    sys : control.StateSpace
        The state-space system describing the dynamics.
    stability_type : str, optional
        The stability type to test for. Can be either 'asymptotical' or 
        'marginal'. Default is asymptotical.

    Returns
    -------
    stable : bool
        True of stable, False if not stable
    poles : pole locations of the system. 
    """
    
    poles = sys.poles()
    
    if stability_type == 'asymptotical':
        stable = np.all(np.real(poles) < 0)
    elif stability_type == 'marginal':
        stable = np.all(np.real(poles) <= 0.0)
    else:
        msg = (f"Unknown stability type {stability_type}! Allowed types are: "
               f"['asymptotical','marginal'].")

    return stable, poles 

def from_gains(A, B, C, D, K_x, K_u=None):
    """
    Create a statespace dynamics object from the list of gains.
    
    Builds a full-state feedback control system.

    Parameters
    ----------
    A : array-like
        System matrix.
    B : array-like
        Input matrix.
    C : array-like
        Output matrix. C has to be shaped (1, n_states) and may have only
        one elements set to 1. All other have to be 0. (MISO-system). Set the
        state that should follow the reference input to one.
    D : array-like
        Feedthrough matrix.
    poles : array-like
        Desired poles of the system.
    t_end : float, optional
        Simulation time for determining the input gain matrix.
        The default is 10.0.
    t_s : TYPE, optional
        Simulation step time for determining the input gain matrix.
        The default is 0.01.

    Returns
    -------
    state_space : control.StateSpace
        Statespace object.

    gains : list
        List of the computed gains that result in poles at the desired
        location. Given as (K_x, K_u).

    """
    
    assert (
        np.linalg.matrix_rank(ct.ctrb(A, B)) == A.shape[0]
    ), "System not controllable!"
    
    if K_u is None: 
        K_u = np.identity(B.shape[1]) * K_x[0,-1]

    assert (K_x.shape[0] == B.shape[1]) and (K_x.shape[1] == A.shape[1]), \
        (f"K_x has to be shaped ({B.shape[1]}, {A.shape[1]}), instead it was " 
         f"{K_x.shape}")
    assert (K_u.shape[0] == B.shape[1]) and (K_u.shape[1] == B.shape[1]), \
        (f"K_u has to be shaped ({B.shape[1]}, {B.shape[1]}), instead it was " 
         f"{K_u.shape}")
    
    return ct.StateSpace(A - B @ K_x, B @ K_u, C, D), (K_x, K_u)

def from_pole_placement(A, B, C, D, poles, t_end=10.0, t_s=0.01):
    """
    Create a statespace dynamics object uing pole placement.
    
    Builds a full-state feedback control system.

    Parameters
    ----------
    A : array-like
        System matrix.
    B : array-like
        Input matrix.
    C : array-like
        Output matrix. C has to be shaped (1, n_states) and may have only
        one elements set to 1. All other have to be 0. (MISO-system). Set the
        state that should follow the reference input to one.
    D : array-like
        Feedthrough matrix.
    poles : array-like
        Desired poles of the system.
    t_end : float, optional
        Simulation time for determining the input gain matrix.
        The default is 10.0.
    t_s : TYPE, optional
        Simulation step time for determining the input gain matrix.
        The default is 0.01.

    Returns
    -------
    state_space : control.StateSpace
        Statespace object.

    gains : list
        List of the computed gains that result in poles at the desired
        location. Given as (K_x, K_u).

    """

    assert (
        np.linalg.matrix_rank(ct.ctrb(A, B)) == A.shape[0]
    ), "System not controllable!"

    K_x = ct.place(A, B, poles)
    K_u = np.identity(B.shape[1])

    sys_temp = ct.ss(A - B @ K_x, B @ K_u, C, D)

    # scale reference to match output
    T = np.arange(t_end, step=t_s)
    U = np.zeros((B.shape[1], T.shape[0]))
    U[:, 10:] = 1
    if B.shape[1] == 1:
        U = U.flatten()
    results = ct.forced_response(sys_temp, T, U, squeeze=False)

    K_u = np.zeros((B.shape[1], B.shape[1]))
    for i in range(B.shape[1]):
        K_u[i, i] = 1 / results.outputs[i, -1]

    gains = (K_x, K_u)
    return ct.StateSpace(A - B @ K_x, B @ K_u, C, D), gains
