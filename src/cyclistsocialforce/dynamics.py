# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 11:55:09 2024

@author: Christoph M. Konrad
"""

import numpy as np
import sympy as sm
import control as ct

from scipy.optimize import root

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

    The dynamics are integrated using the midpoint method. Scipy.optimize.root
    solves the non-linear equations of motion at each step.

    Note: The full midpoint integration method would require the input to the 
    dynamics (i.e. the resulting force) at time u(t+h/2). 
    I.e. Fx(t+t/2) = Fx[n] + Fx[n+1] / 2. However, Fx[n+1] is not known at time n.
    Including it as a state into the dynamics is not possible because it's calculation
    requires the states of the other road users in the evironment. This would
    effectively link the dynamics of all existing roaduser which we avoid to 
    manage the complexity of the problem. Instead we assume 
    u(t+h/2) = u(t) for each step and accept a small error. 
    
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
        self._config_gains_and_poles(bicycle.params)
            
        # get geometry parameters
        w = self.bp_model.parameter_set.parameters["w"]
        c = self.bp_model.parameter_set.parameters["c"]
        coslam = np.cos(self.bp_model.parameter_set.parameters["lam"])

        # pre-calc yaw state matrix coefficients
        self.A41_over_v = coslam / w
        self.A43 = coslam * c / w

        # parameters
        self.t_s = bicycle.params.t_s
        self.a_max = bicycle.params.a_max
        self.v_max = bicycle.params.v_max_riding

        # initialize state space system
        self.x, self.v = self._transform_state_csf2dynamics(bicycle.s)
        self.gains = self._get_gains(self.v)

        # initialize lateral dynamics integrators
        self.eval_lat_residual, self.eval_lat_jacobian = self._get_midpoint_moment_evaluators()

        # initialize trivial speed controller
        self.speed_controller = PIDcontroller(
            bicycle.params.k_p_v, 0, 0, bicycle.params.t_s, isangle=False
        )

        # roll and steer torque disturbance
        if bicycle.params.p_dist_roll > 0 or bicycle.params.p_dist_steer:
            raise Warning("Support for steer and roll torque disturbance removed!")
        

    def _transform_state_dynamics2csf(self, state_dyn, speed):
        """ Transforms the state of the dynamic system to cyclistsocialforce coordinates. Ensures that
        angles are withtin [-pi, pi].

        CSF coordinates (E-frame):
            ex - forward
            ey - left
            ez - up

        Bikemodel coordinates (N-frame), as defined by Moore (2012) and Meijaard et al. (2007)
            nx - forward
            ny - right
            ny - down
        
        Parameters
        ----------
        state_dyn : array-like
            The state vector in the bikemodel coordinates: [roll, steer, roll-rate, steer-rate, yaw, x, y]

        Returns
        -------
        state_csf : np.ndarray
            The state vector in cyclistsocialforce coordinates: [x, y, yaw, speed, steer, roll, steer-rate, roll-rate]
        """

        # steer(rate), yaw and y need to be mirrored
        state_cfs = np.array([
            state_dyn[5],               # x
            -state_dyn[6],              # y
            -limitAngle(state_dyn[4]),  # yaw
            speed,                      # speed
            -limitAngle(state_dyn[1]),  # steer
            limitAngle(state_dyn[0]),   # roll
            -state_dyn[3],               # steer-rate
            state_dyn[2],              # roll-rate
        ])

        return state_cfs


    def _transform_state_csf2dynamics(self, state_csf):
        """ Transforms the vehicle state given in cyclistsocialforce coordinates
        into the coordinates of the dynamic system. 

        CSF coordinates (E-frame):
            ex - forward
            ey - left
            ez - up

        Bikemodel coordinates (N-frame), as defined by Moore (2012) and Meijaard et al. (2007)
            nx - forward
            ny - right
            ny - down
        
        Parameters
        ----------
        state_csf : array-like
            The state vector in cyclistsocialforce coordinates: [x, y, yaw, speed, steer, roll, steer-rate, roll-rate]

        Returns
        -------
        state_dyn : np.ndarray
            The state vector in the bikemodel coordinates: [roll, steer, roll-rate, steer-rate, yaw, x, y]
        speed : float
            The forward speed (same in both reference systems)
        """

        # steer(rate), yaw and y need to be mirrored
        state_dyn = np.array([
            state_csf[5],   #roll 
            -state_csf[4],  #steer 
            state_csf[7],   #roll-rate
            -state_csf[6],  #steer-rate 
            -state_csf[2],  #yaw
            state_csf[0],   #x 
            -state_csf[1]   #y
        ])

        return state_dyn, state_csf[3]
        

    def _config_gains_and_poles(self, params):
        """ Configure if dynamics are defined by desired gains or desired poles. Desired poles overwrite desired gains.
        """
        
        self.from_gains = False
        self.desired_gains = None
        self.desired_poles = None

        if hasattr(params, 'gains'):
            if not params.gains is None:
                self.from_gains = True
                self.desired_gains = params.gains
        
        if hasattr(params, 'poles'):
            if not params.poles is None:
                self.from_gains = False
                self.desired_poles = params.poles

        if (self.desired_gains is None) and (self.desired_poles is None):
            msg = ("The BicycleParameter object neither defines desired gains nor desired poles! Make sure that"
                   "'params' has a 'gains' or a 'poles' attribute and that at least one of them is not 'None'.")
            raise RuntimeError(msg)
        

    def _make_eoms_set_whipplecarvallo(self):
        """
        Returns dx(t)/dt = f(x(t), u(t)) with

        States and input:
        x(t) = [phi(t), delta(t), dphi(t)/dt, ddelta(t)/dt, psi(t), p_x(t), p_y(t)]^T
        u(t) = [p_x_c(t), p_y_c(t)]^T

        Unknown gain parameters:
        k = [k_phi, k_delta, k_phidot, k_deltadot, k_psi]

        State derivative:
        dx(t)/dt = [dphi(t)/d, ddelta(t)/d, ddphi(t)/dt^2, dddelta(t)/dt^2, dpsi(t)/dt, dp_x(t)/dt, dp_y(t)/dt]^T

        Returns
        -------
        f : sm.Matrix
            The function f(x(t), u(t)). Shaped [7,1]
        states : sm.Matrix
            The state vector shaped [7,1]
        params : sm.Matrix
            A vector of the unknown gain parameters in f.
        inputs : sm.Matrix
            The input vector u(t) shaped [2,1].
        v : sm.Function
            The unknown speed.
        
        """

        # input symbols
        psi_c, v = sm.symbols("psi_c, v")
        
        # state symbols
        state_names = ["phi", "delta", "phidot", "deltadot", "psi", "p_x", "p_y"]
        param_names = ["k_phi", "k_delta", "k_phidot", "k_deltadot", "k_psi"]

        # ids of position and orientation states
        pos_state_ids = np.array([5,6])
        psi_state_id = 4
        bikerider_state_ids = np.array([0,1,2,3,4])
        Kx_param_ids = np.array([0,1,2,3,4])
        Ku_param_ids = np.array([4])

        inputs = [psi_c]
        states = [sm.Symbol(s) for s in state_names]
        params = [sm.Symbol(p) for p in param_names]

        # bike-rider states
        x_br = sm.Matrix([[states[j]] for j in bikerider_state_ids])
    
        # bike-rider state-space matrices
        A_br, B_br, _, _ = self.get_symbolic_statespace_matrices()
        A_br = sm.Matrix(A_br)
        B_br = sm.Matrix(B_br[:, 1])  # only use steer torque input
    
        # gain matrices
        K_x = sm.Matrix([[params[j]] for j in Kx_param_ids]).T
        K_u = sm.Matrix([[params[j]] for j in Ku_param_ids])
        
        # bike-rider eoms
        f_br = ((A_br - B_br * K_x) * x_br + B_br * K_u * psi_c)
        
        # forward motion eoms 
        f_fw = sm.Matrix(
            [[v * sm.cos(states[psi_state_id])],   #xdot - v * cos(psi)
            [v * sm.sin(states[psi_state_id])]])  #ydot - v * sin(psi)
    
        # combine bike-rider eoms and forward eoms
        f = f_br.col_join(f_fw)

        return f, states, params, inputs, v


    def _get_midpoint_moment_evaluators(self):

        f, states, params, inputs, v = self._make_eoms_set_whipplecarvallo()

        x = sm.Matrix([sm.Symbol(f"{s.name}_n") for s in states])
        x_next = sm.Matrix([sm.Symbol(f"{s.name}_(n+1)") for s in states])

        x_repl = dict(zip(states, (x+x_next)/2))

        h = sm.symbols("h")

        # Residual R of the implicit midpoint method and it's jacobian.
        R = x_next - x - h * f.subs(x_repl)
        J = R.jacobian(x_next)

        eval_residual = sm.lambdify((params, inputs, x, x_next, v, h), R)
        eval_jacobian = sm.lambdify((params, inputs, x, x_next, v, h), J)

        return eval_residual, eval_jacobian
    

    def get_statespace_matrices(self, v):
        """
        xdot = A*x + B*u
        
        with 
        
        x = [phi, delta, phidot, deltadot, psi]^T
        u = [0, Tdelta]

        """
        
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
    
    def get_symbolic_statespace_matrices(self, t=None):
        """
        Return the statespace matrices in symbolic form depending on the 
        speed v(t). 
        
        dx = A(v(t))x + Bu
        y = Cx + Du
        
        for x = [phi, delta, dphi, ddelta, psi]^T
        and u = [Tphi, Tdelta]^T

        A(v(t))

        Parameters
        ----------
        t : Sympy.Symbol, optional
            Time symbol. If provided, The state-space matrices will include v(t) as sm.Function. 
            Otherwise, v will be included as sm.Symbol

        Returns
        -------
        A : Sympy.matrix
            Matrix A(v(t)).
        B : Numpy.ndarray
            Matrix B
        C : Numpy.ndarray
            Matrix C
        D : Numpy.ndarray
            Matrix D

        """

        M, C1, K0, K2 = self.bp_model.form_reduced_canonical_matrices()
        g = 9.81

        if t is not None:
            v = sm.Function("v")(t)
        else:
            v = sm.Symbol("v")
        
        Minv = np.linalg.inv(M)

        # A matrix
        A = sm.Matrix(np.zeros((5,5)))
        A[2:4, 0:2] = -Minv @ (g * K0 + v**2 * K2)
        A[0:2, 2:4] = np.identity(2)
        A[2:4, 2:4] = -Minv @ C1 * v
        A[4, 1] = self.A41_over_v * v
        A[4, 3] = self.A43

        # B matrix
        B = np.zeros((5, 2))
        B[2:4, :] = Minv

        # output
        C = np.zeros((1, A.shape[1]))
        C[0, 4] = 1
        D = np.zeros((C.shape[0], B.shape[1]))

        return A, B, C, D
    

    def _get_gains(self, v):
        """ Calculates the gains at a given speed.
        """

        if self.from_gains:
            return self.desired_gains
        else:
            A, B, C, D = self.get_statespace_matrices(v)
            _, gains = from_pole_placement(
                A, B[:, 1][:, np.newaxis], C, D[:, 1][:, np.newaxis],
                self.desired_poles)
            
            return gains[0]


    def _step_speed(self, bicycle, Fx, Fy):
        """Propagate the speed dynamics by one time step.

        Parameters
        ----------
        bicycle : cyclistsocialforce.Bicycle
            Bicycle this speed step is associated with. Unused, kept for API
            consistency.
        Fx : float
            X-component of the current social force.
        Fy : float
            y-component of the current social force.
        a_max

        Returns
        -------
        v : float
            Updated speed

        """

        # desired speed
        vd = np.sqrt(Fx**2 + Fy**2)

        # acceleration
        a = self.speed_controller.step(vd - self.v)
        a = thresh(a, self.a_max)

        # integrate speed
        v = thresh(self.v + self.t_s * a, self.v_max)

        return v
    

    def _calc_commanded_yaw(self, Fx, Fy):
        """ Calculate the commanded yaw angle from the social forces.

        The yaw angle is normally represented in [-np.pi, pi]. To make sure that the
        correct rotation is chosen to compensate the commanded yaw, the commanded
        angle needs to be augmented to the interval [psi-np.pi, psi+pi].

        """
        
        #transform lateral force to bikemodel reference frame
        Fy = - Fy

        psi_F = limitAngle(np.arctan2(Fy, Fx))

        psi = self.x[4]
        delta_psi = angleDifference(psi, psi_F) 

        psi_c = psi + delta_psi

        return psi_c


    def step(self, bicycle, Fx, Fy):

        #update speed
        v = self._step_speed(bicycle, Fx, Fy)

        #update gains if speed changed
        if v != self.v:
            self.gains = self._get_gains((v+bicycle.s[3])/2)
        gains = self.gains.flatten()

        #inputs, assuming u(t+h/2) = u(t)
        psi_c = self._calc_commanded_yaw(Fx, Fy)
        inputs = np.array([psi_c])

        #integration of lateral dynamics

        def residual(x_next):
            return self.eval_lat_residual(gains, inputs, self.x, x_next, (v+bicycle.s[3])/2, self.t_s).flatten()

        def jacobian(x_next):
            return self.eval_lat_jacobian(gains, inputs, self.x, x_next, (v+bicycle.s[3])/2, self.t_s)
        
        sol = root(residual, self.x, jac=jacobian, method='lm')
        if not sol.success:
            raise RuntimeError(f"Failed to solve nonlinear dynamics of agent {bicycle.id} in step {bicycle.i}! scipy.optimize.root exited with '{sol.message}'")

        # update state
        self.x = sol.x
        self.v = v

        # update bicycle state
        bicycle.s = self._transform_state_dynamics2csf(self.x, self.v)
        

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


class ParticleDynamics(Dynamics):
    """ A class for modelling the dynamics of a bicycle as a mass-less 
    particle in planar space"""


    def __init__(self, vehicle):

        super().__init__(vehicle)

        self.eval_lat_residual, self.eval_lat_jacobian = self._get_midpoint_moment_evaluators()
        self._config_gains_and_poles(vehicle.params)

        # parameters
        self.t_s = vehicle.params.t_s
        self.a_max = vehicle.params.a_max
        self.v_max = vehicle.params.v_max_riding

        # set fixed gains for the given design speed. 
        self.gains = self._get_gains()

        # initialize trivial speed controller
        self.speed_controller = PIDcontroller(
            vehicle.params.k_p_v, 0, 0, vehicle.params.t_s, isangle=False
        )


        self.x, self.v = self._transform_state_csf2dynamics(vehicle.s)


    def _config_gains_and_poles(self, params):
        """ Configure if dynamics are defined by desired gains or desired poles. Desired poles overwrite desired gains.
        """
        
        self.from_gains = False
        self.desired_gains = None
        self.desired_poles = None

        if hasattr(params, 'gains'):
            if not params.gains is None:
                self.from_gains = True
                self.desired_gains = params.gains
        
        if hasattr(params, 'poles'):
            if not params.poles is None:
                self.from_gains = False
                self.desired_poles = params.poles

        if (self.desired_gains is None) and (self.desired_poles is None):
            msg = ("The VehicleParameter object neither defines desired gains nor desired poles! Make sure that"
                   "'params' has a 'gains' or a 'poles' attribute and that at least one of them is not 'None'.")
            raise RuntimeError(msg)


    def _make_eom_set_particle(self):
        """
        Returns dx(t)/dt = f(x(t), u(t)) with

        States and input:
        x(t) = [psi(t), p_x(t), p_y(t)]^T
        u(t) = [p_x_c(t), p_y_c(t)]^T

        Unknown gain parameters:
        k = [k_psi]

        State derivative:
        dx(t)/dt = [dpsi(t)/dt, dp_x(t)/dt, dp_y(t)/dt]^T

        Returns
        -------
        f : sm.Matrix
            The function f(x(t), u(t)). Shaped [3,1]
        states : sm.Matrix
            The state vector shaped [3,1]
        params : sm.Matrix
            A vector of the unknown gain parameters in f.
        inputs : sm.Matrix
            The input vector u(t) shaped [3,1].
        v : sm.Function
            The unknown speed.
        """

        # input symbols
        psi_c, v = sm.symbols("psi_c, v")
        inputs = [psi_c]

        # ids of position and orientation states
        pos_state_ids = np.array([1,2])
        psi_state_id = 0

        # states and parameters
        state_names = ["psi", "p_x", "p_y"]
        param_names = ["k_psi"]

        states = [sm.Symbol(s) for s in state_names]
        params = [sm.Symbol(p) for p in param_names]
    
        # eoms
        f_psi = sm.Matrix(
            [(- params[0] * (states[psi_state_id] - psi_c))])

        # forward motion eoms 
        f_fw = sm.Matrix([[v * sm.cos(states[psi_state_id])],   #v * cos(psi)
                          [v * sm.sin(states[psi_state_id])]])  #v * sin(psi)
        
        # combine yaw_tracking and forward eoms
        f = f_psi.col_join(f_fw)

        return f, states, params, inputs, v


    def _get_midpoint_moment_evaluators(self):

        f, states, params, inputs, v = self._make_eom_set_particle()

        x = sm.Matrix([sm.Symbol(f"{s.name}_n") for s in states])
        x_next = sm.Matrix([sm.Symbol(f"{s.name}_(n+1)") for s in states])

        x_repl = dict(zip(states, (x+x_next)/2))

        h = sm.symbols("h")

        # Residual R of the implicit midpoint method and it's jacobian.
        R = x_next - x - h * f.subs(x_repl)
        J = R.jacobian(x_next)

        eval_residual = sm.lambdify((params, inputs, x, x_next, v, h), R)
        eval_jacobian = sm.lambdify((params, inputs, x, x_next, v, h), J)

        return eval_residual, eval_jacobian


    def _get_gains(self):
        """ Calculates the gains.
        """

        if self.from_gains:
            return self.desired_gains
        else:
            return -np.real(self.desired_poles)
        

    def _transform_state_dynamics2csf(self, state_dyn, speed):
        """ Transforms the state of the dynamic system to cyclistsocialforce coordinates. Ensures that
        angles are withtin [-pi, pi].
        
        Parameters
        ----------
        state_dyn : array-like
            The state vector in the bikemodel coordinates: [yaw, x, y]

        Returns
        -------
        state_csf : np.ndarray
            The state vector in cyclistsocialforce coordinates: [x, y, yaw, speed]
        """

        # steer(rate), yaw and y need to be mirrored
        state_cfs = np.array([
            state_dyn[1],              # x
            state_dyn[2],              # y
            limitAngle(state_dyn[0]),  # yaw
            speed,                     # speed
        ])

        return state_cfs


    def _transform_state_csf2dynamics(self, state_csf):
        """ Transforms the vehicle state given in cyclistsocialforce coordinates
        into the coordinates of the dynamic system. 
        
        Parameters
        ----------
        state_csf : array-like
            The state vector in cyclistsocialforce coordinates: [x, y, yaw, speed]

        Returns
        -------
        state_dyn : np.ndarray
            The state vector in the bikemodel coordinates: [yaw, x, y]
        speed : float
            The forward speed (same in both reference systems)
        """

        # steer(rate), yaw and y need to be mirrored
        state_dyn = np.array([
            state_csf[2],  #yaw
            state_csf[0],  #x 
            state_csf[1]   #y
        ])

        return state_dyn, state_csf[3]
    
    
    def _calc_commanded_yaw(self, Fx, Fy):
        """ Calculate the commanded yaw angle from the social forces.

        The yaw angle is normally represented in [-np.pi, pi]. To make sure that the
        correct rotation is chosen to compensate the commanded yaw, the commanded
        angle needs to be augmented to the interval [psi-np.pi, psi+pi].

        """
        psi_F = limitAngle(np.arctan2(Fy, Fx))

        psi = self.x[0]
        delta_psi = angleDifference(psi, psi_F) 

        psi_c = psi + delta_psi

        return psi_c


    def _step_speed(self, vehicle, Fx, Fy):
        """Propagate the speed dynamics by one time step.

        Parameters
        ----------
        vehicle : cyclistsocialforce.vehicle
            Vehicle this speed step is associated with. Unused, kept for API
            consistency.
        Fx : float
            X-component of the current social force.
        Fy : float
            y-component of the current social force.
        a_max

        Returns
        -------
        v : float
            Updated speed

        """

        # desired speed
        vd = np.sqrt(Fx**2 + Fy**2)

        # acceleration
        a = self.speed_controller.step(vd - self.v)
        a = thresh(a, self.a_max)

        # integrate speed
        v = thresh(self.v + self.t_s * a, self.v_max)

        return v


    def step(self, vehicle, Fx, Fy):

        #update speed
        v = self._step_speed(vehicle, Fx, Fy)

        #gains
        gains = self.gains.flatten()

        #inputs, assuming u(t+h/2) = u(t)
        psi_c = self._calc_commanded_yaw(Fx, Fy)
        inputs = np.array([psi_c])


        def residual(x_next):
            return self.eval_lat_residual(gains, inputs, self.x, x_next,  (v+vehicle.s[3])/2, self.t_s).flatten()

        def jacobian(x_next):
            return self.eval_lat_jacobian(gains, inputs, self.x, x_next,  (v+vehicle.s[3])/2, self.t_s)
        
        sol = root(residual, self.x, jac=jacobian, method='lm')
        if not sol.success:
            raise RuntimeError(f"Failed to solve nonlinear dynamics of agent {vehicle.id} in step {vehicle.i}! scipy.optimize.root exited with '{sol.message}'")

        # update state
        self.x = sol.x
        self.v = v

        # update bicycle state
        vehicle.s = self._transform_state_dynamics2csf(self.x, self.v)
        
        
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
