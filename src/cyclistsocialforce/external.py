# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 10:08:50 2024

Implementations and interfaces of external models to be used in the
cyclistsocialforce framework.

Implemented models:
    - KATHS Cyclist Model

@author: Christoph M. Konrad
"""

import numpy as np
from cyclistsocialforce.vehicle import Vehicle
from cyclistsocialforce.utils import limitAngle
from cyclistsocialforce.vizualisation import (
    CarDrawing2D,
    BicycleDrawing2D,
)
from cyclistsocialforce.parameters import BicycleParameters, CarParameters


""" KATHS Cyclist Model -------------------------------------------------------

Particle-based cyclist model by Kaths (2023)

Implementation by Christoph M. Konrad based solely on information from the
associated publication. The implementation was not verified by the original
author of the model. 

Different to the original paper, this implementation does not include the
stopping-at-traffic-lights feature and has deterministic parameters. 

TODO: implement stochastic parameters.

Kaths H (2023), A movement and interaction model for cyclists and other 
non-lane-based road users. Front. Future Transp. 4:1183270. 
doi: 10.3389/ffutr.2023.1183270

"""


def step_kaths_particle_model(Vehicle, Fv, Ft):
    psi = limitAngle(Ft * Vehicle.params.t_s + Vehicle.s[2])

    v = Vehicle.s[3] + Vehicle.params.t_s * Fv
    y = Vehicle.s[1] + Vehicle.params.t_s * v * np.sin(psi)
    x = Vehicle.s[0] + Vehicle.params.t_s * v * np.cos(psi)

    Vehicle.s[:4] = (x, y, psi, v)


def get_kaths_veloaniso_paramset(draw_random=False):
    params = {
        "A_tb": 0.48,
        "R_vb": 3.10,
        "R_tb": 1.91,
        "gamma_tb": 0.97,
        "gamma_vb": 1.03,
        "eta_vb": 2.05,
        "eta_tb": 1.96,
        "T_vb": 2.05,
        "T_tb": 1.15,
    }

    return params


def calc_kaths_veloaniso_destination_force(Vehicle):
    """Implements the destination force of the velocity anisotropic model by
    Heather Kaths (2023) (DOI: 10.3389/ffutr.2023.1183270).

    """

    t_b_0 = np.arctan(
        (Vehicle.dest[1] - Vehicle.s[1]) / (Vehicle.dest[0] - Vehicle.s[0])
    )

    Fv = (
        Vehicle.params.v_desired_default - Vehicle.s[3]
    ) / Vehicle.params.dest_force["T_vb"]
    Ft = (t_b_0 - Vehicle.s[2]) / Vehicle.params.dest_force["T_tb"]
    
    return Fv, Ft


def calc_kaths_veloaniso_repulsive_force(Vehicle, x, y, psi):
    """
    Implements the velocity anisotropic repulsive forces of the cyclist
    model by Heather Kaths (DOI: 10.3389/ffutr.2023.1183270).

    Does not implement the stop line functionality


    Returns
    -------
    None.

    """

    A_vb = (
        Vehicle.params.v_desired_default
        + (Vehicle.params.rep_force["T_vb"] - 1) * Vehicle.s[3]
    ) / Vehicle.params.rep_force["T_vb"]

    d_bi = np.array((x, y)) - Vehicle.s[0:2]
    D_bi = np.sqrt(d_bi[0, :] ** 2 + d_bi[:, 1] ** 2)

    e_vb = np.array((np.cos(Vehicle.s[2]), np.cos(Vehicle.s[2])))
    e_wb = np.flip(e_vb)
    e_vi = np.array((np.cos(psi), np.sin(psi)))

    D_vbi_1star = d_bi @ e_vb + Vehicle.params.rep_force["eta_vb"] * (
        d_bi @ e_wb
    )
    D_tbi_1star = d_bi @ e_vb + Vehicle.params.rep_force["eta_vb"] * (
        d_bi @ e_wb
    )
    D_vbi_2star = D_vbi_1star + Vehicle.params.rep_force["gamma_vb"] * (
        e_vb @ e_vi
    )
    D_tbi_2star = D_tbi_1star + Vehicle.params.rep_force["gamma_tb"] * (
        e_vb @ e_vi
    )

    # Simplified to give the sign of the z-component of the normal.
    # Equivalent to eq.8 in Kaths (2023)
    U_bi = np.sign(np.cross(e_vb, d_bi, axisb=0))

    Fv = -A_vb * np.exp(
        -np.min(D_vbi_2star) / Vehicle.params.rep_force["R_vb"]
    )
    Ft = -Vehicle.params.rep_force["A_tb"] * np.sum(
        U_bi * np.exp(-D_tbi_2star / Vehicle.params.rep_force["R_tb"])
    )

    return Fv, Ft


class Kaths_Vehicle(Vehicle):
    def __init__(self, s0, **kwargs):
        kwargs = dict(
            kwargs,
            dyn_step_func=step_kaths_particle_model,
            rep_force_func=calc_kaths_veloaniso_repulsive_force,
            dest_force_func=calc_kaths_veloaniso_destination_force,
        )

        super().__init__(s0, **kwargs)


class Kaths_Bicycle(Kaths_Vehicle):
    def __init__(self, s0, v_desired_default, **kwargs):
        kaths_params = get_kaths_veloaniso_paramset()
        params = BicycleParameters(
            v_desired_default=v_desired_default,
            rep_force=kaths_params,
            dest_force=kaths_params,
        )

        kwargs = dict(kwargs, params=params, drawing_class=BicycleDrawing2D)

        super().__init__(s0, **kwargs)


class Kaths_Car(Kaths_Vehicle):
    def __init__(self, s0, traj, **kwargs):
        kaths_params = get_kaths_veloaniso_paramset()
        params = CarParameters(rep_force=kaths_params, dest_force=kaths_params)
        params.v_desired_default = 10

        kwargs = dict(
            kwargs,
            params=params,
            drawing_class=CarDrawing2D,
            uncontrolled=True,
            uncontrolled_traj=traj,
        )

        super().__init__(s0, **kwargs)
