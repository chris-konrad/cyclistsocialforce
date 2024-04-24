# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 11:55:09 2024

@author: Christoph M. Schmidt
"""

import numpy as np
import control as ct
import bicycleparameters as bp

from cyclistsocialforce.utils import limitAngle, thresh
from cyclistsocialforce.vehiclecontrol import PIDcontroller

class Dynamics():
    
    def __init__(self, Vehicle):
        self.x = Vehicle.s
    
    def step(self, Vehicle, F1, F2):
        pass
    
    
    
class WhippleCarvalloDynamics(Dynamics):
    
    PATH = "U:\PhDConnectedVRU\Projects\external\BicycleParameters\data"
    BIKE = "Browser"
    
    def __init__(self, bicycle, poles = (-18 + 0j, -19 + 0, -20 + 0j, -6 + 3j, -6 - 3j)):
        
        self.poles = poles
        
        #get transition and input matrices from Jason Moore's toolbox
        self.bpbike = bp.Bicycle(self.BIKE, pathToData=self.PATH)
        
        #get geometry parameters
        params = bp.io.remove_uncertainties(self.bpbike.parameters['Benchmark'])
        self.l = params['w']
        
        #initialize state space system
        self.update(bicycle.s[3])
        
        #save initial state
        self.x = np.zeros(5)
        self.x[0] = bicycle.s[4]
        self.x[2] = bicycle.s[5] 
        self.x[4] = bicycle.s[2]
        
        #speed controller
        self.speed_controller = PIDcontroller(
            bicycle.params.k_p_v, 0, 0, bicycle.params.t_s, isangle=False
        )
        
    def update(self, v):
        Awc, Bwc = self.bpbike.state_space(v, nominal=True)
        
        #add yaw dynamics
        A = np.zeros((5,5))
        A[:4,:4] = Awc
        A[4,0] = v / self.l
        
        B = np.zeros((5,1))
        B[:4,0] = Bwc[:,1]  #disregard roll torque input
        

        
        #output
        C = np.array([[1,0,1,0,1]])
        D = np.zeros((C.shape[0], B.shape[1]))
    
        #pole placement
        self.sys = from_pole_placement(A, B, C, D, self.poles)
        
        
    def step(self, bicycle, Fx, Fy):
        
        # update statespace parameters with current speed
        self.update(np.sqrt(Fx**2+Fy**2))

        # absolute force angle
        psi_d = np.arctan2(Fy, Fx)

        # calculate steer angle for stabilization
        results = ct.forced_response(
            self.sys,
            T=np.array([0, bicycle.params.t_s]),
            X0=self.x,
            U=np.ones(2) * psi_d,
            squeeze=False,
        )
        self.x = results.outputs[:,1]
        
        bicycle.s[2] = limitAngle(results.outputs[4,1])
        bicycle.s[4] = limitAngle(results.outputs[0,1])
        bicycle.s[5] = limitAngle(results.outputs[2,1])
        
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
        
        
class ParticleDynamicsXY(Dynamics):
    
    def __init__(self, Vehicle, poles = (0 + 0j, 0 + 0j, -3 + 0j, -3 + 0j)):
        
        # init state space system
        A = np.array([[0,0,1,0], [0,0,0,1], [0,0,0,0], [0,0,0,0]])
        B = np.array(([[0,0],[0,0],[1,0],[0,1]]))
        C = np.array([[0,0,1,0], [0,0,0,1]])
        D = np.zeros((C.shape[0], B.shape[1]))
        
        self.sys = from_pole_placement(A, B, C, D, poles)
        
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
            U=np.ones(2) * np.array(((Fx,),(Fy,))),
            squeeze=False)
        
        self.x = results.outputs[:,1]
        
        Vehicle.s[0:2] = results.outputs[0:2,1]
        Vehicle.s[2] = np.arctan((results.outputs[1,1] - results.outputs[1,0])/
                                 (results.outputs[0,1] - results.outputs[0,0]))
        Vehicle.s[3] = np.sqrt(np.sum(results.outputs[2:4,1]**2))
        
        
def from_pole_placement(A, B, C, D, poles, t_end=10, t_s=0.01):


    assert (
        np.linalg.matrix_rank(ct.ctrb(A, B)) == A.shape[0]
    ), "System not controllable!"
    
    K_x = ct.place(A, B, poles)
    K_u = np.identity(B.shape[1])

    sys_temp = ct.ss(A - B @ K_x, B @ K_u , C, D)

    # scale reference to match output
    T = np.arange(t_end, step=t_s)
    U = np.zeros((B.shape[1],T.shape[0]))
    U[:,10:] = 1
    if B.shape[1] == 1:
        U = U.flatten()
    results = ct.forced_response(sys_temp, T, U, squeeze=False)
    
    K_u = np.zeros((B.shape[1], B.shape[1]))
    for i in range(B.shape[1]):
        K_u[i,i] = 1/results.outputs[i,-1]

    # final controlled system
    C = np.identity(A.shape[0])
    D = np.zeros((C.shape[0], B.shape[1]))
    
    return ct.StateSpace(A - B @ K_x, B @ K_u, C, D)
        
        