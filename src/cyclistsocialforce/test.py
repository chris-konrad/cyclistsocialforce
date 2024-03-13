# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 09:36:53 2024

@author: Christoph M. Schmidt
"""
import control as ct
import matplotlib.pyplot as plt
import numpy as np
import unittest

from cyclistsocialforce.vehicle import InvPendulumBicycle


class TestBicycleDynamics(unittest.TestCase):
    def test_yaw_stepresponse_invpend(self):
        """
        Tests the isolated yaw dynamics of the inverted pendulum bicycle
        implementation.

        Speed is kept constant.

        Returns
        -------
        None.

        """

        s0 = [0, 0, 0, 0, 0, 0]
        testbike = InvPendulumBicycle(s0)
        testbike.s[3] = testbike.params.v_desired_default

        # test input: 30 deg yaw step, 10 s sim time
        t = np.arange(0, 10, testbike.params.t_s)
        psi_d = np.zeros_like(t)
        psi_d[int(0.2 * len(t)) :] = 2 * np.pi * 30 / 360

        Fx = testbike.params.v_desired_default * np.cos(psi_d)
        Fy = testbike.params.v_desired_default * np.sin(psi_d)

        # run test on package implementation
        for fx, fy in zip(Fx, Fy):
            testbike.s[[2, 4, 5]] = testbike.step_yaw(fx, fy)

            # counter and trajectories.
            testbike.i += 1
            testbike.i = testbike.i % testbike.traj.shape[1]

            testbike.traj[:, testbike.i] = testbike.s

        # run test on reference implementation
        v = testbike.params.v_desired_default
        K = v**2 / (testbike.params.g * testbike.params.l)
        tau_2 = testbike.params.l_2 / v
        tau_3 = testbike.params.l / v

        A = np.array(
            [
                [0, 1, 0, 0, 0],
                [
                    0,
                    -testbike.params.c_steer
                    / testbike.params.i_steer_vertvert,
                    0,
                    0,
                    0,
                ],
                [0, 0, 0, 1, 0],
                [
                    -K / testbike.params.tau_1_squared,
                    -K * tau_2 / testbike.params.tau_1_squared,
                    1 / testbike.params.tau_1_squared,
                    0,
                    0.0,
                ],
                [1 / tau_3, 0, 0, 0, 0],
            ]
        )

        B = np.array([0, 1 / testbike.params.i_steer_vertvert, 0, 0, 0])
        Cpsi = np.array([0, 0, 0, 0, 1])
        D = 0
        Ku = 1 / -0.46313878281084603

        G = ct.ss(A, B, Cpsi, D)

        poles_inner = (
            np.array(
                (-0.2 + 0j, -0.1 + 0.1j, -0.1 - 0.1j, -0.15 + 0j, -0.1 + 0j)
            )
            * 30
        )
        K_inner = ct.place(G.A, G.B, poles_inner)
        G_controlled = ct.ss(A - B[:, np.newaxis] @ K_inner, Ku * B, Cpsi, D)

        t, theta, x = ct.forced_response(
            G_controlled, T=t, squeeze=False, U=psi_d, return_x=True
        )

        # check results
        try:
            np.testing.assert_allclose(
                testbike.traj[4, : len(t)],
                x[0, : len(t)],
                err_msg="Error in steer angle!",
                verbose=False,
            )
            np.testing.assert_allclose(
                testbike.traj[5, : len(t)],
                x[2, : len(t)],
                err_msg="Error in roll angle!",
                verbose=False,
            )
            np.testing.assert_allclose(
                testbike.traj[2, : len(t)],
                x[4, : len(t)],
                err_msg="Error in yaw angle!",
                verbose=False,
            )
        except Exception as e:
            if plot_on_error:
                fig, ax = plt.subplots(3, 1, sharex=True)
                ax[0].set_title("Error in yaw dynamics test!")

                ax[0].plot(t, x[0, :])
                ax[0].plot(t, testbike.traj[4, : len(t)])
                ax[0].set_ylabel("steer angle")
                ax[1].plot(t, x[2, :])
                ax[1].plot(t, testbike.traj[5, : len(t)])
                ax[1].set_ylabel("roll angle")
                ax[2].plot(t, x[4, :], label="reference")
                ax[2].plot(
                    t, testbike.traj[2, : len(t)], label="implementation"
                )
                ax[2].plot(t, psi_d, "k", label="input")
                ax[2].set_ylabel("yaw angle")
                plt.legend()

                # print some excerpts of the vehicle state
                K_x, K_u = testbike.params.fullstate_feedback_gains(
                    testbike.s[3]
                )
                print("-- IMPLEMENTATION --")
                print(f"speed: {testbike.s[3]}")
                print(f"dynamics: ")
                print(f"    A = ")
                print(f"{testbike.dynamics.A}")
                print(f"    B = ")
                print(f"{testbike.dynamics.B}")
                print(f"gains: ")
                print(f"    K_x = {K_x}")
                print(f"    K_u = {K_u}")

                print("-- Reference --")
                print(f"speed: {testbike.params.v_desired_default}")
                print(f"dynamics: ")
                print(f"    A = ")
                print(f"{G_controlled.A}")
                print(f"    B = ")
                print(f"{G_controlled.B}")
                print(f"gains: ")
                print(f"    K_x = {K_inner}")
                print(f"    K_u = {Ku}")

            raise e


if __name__ == "__main__":
    plot_on_error = True
    unittest.main()
