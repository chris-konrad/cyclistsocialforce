# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 16:49:49 2023.

@author: Christoph Schmidt
"""

import numpy as np
import io


def limitMagnitude(x, y, r):
    """Limit the magnitude of a set of vectors to a given maximum



    Parameters
    ----------
    x : numpy.ndarray
        x dimension of the vectors.
    y : numpy.ndarray
        y dimension of the vectors.
    r : numpy.ndarray
        maximum magnitude of each vector.

    Returns
    -------
    x : numpy.ndarray
        rescaled x dimensions
    y : numpy.ndarray
        rescaled x dimensions

    """

    rin = np.sqrt(x**2 + y**2)
    ids = rin > r

    if np.any(rin):
        x[ids] = x[ids] * r[ids] / rin[ids]
        y[ids] = y[ids] * r[ids] / rin[ids]

    return x, y


def figToImg(fig):
    """
    Based on Jonan Gueorguiev and dizcza (https://stackoverflow.com/questions/7821518/save-plot-to-numpy-array)
    """
    with io.BytesIO() as buff:
        fig.savefig(buff, format="raw")
        buff.seek(0)
        data = np.frombuffer(buff.getvalue(), dtype=np.uint8)
    w, h = fig.canvas.get_width_height()
    return data.reshape((int(h), int(w), -1))


def toDeg(rad):
    return 360 * rad / (2 * np.pi)


def toRad(deg):
    return 2 * np.pi * deg / (360)


def clearAxes(ax):
    for e in ax.get_children():
        e.remove()


def angleSUMOtoSFM(theta):
    """Convert angle between SUMO definition and SFM definition"""
    return limitAngle((np.pi / 2) - toRad(theta))


def angleSFMtoSUMO(theta):
    """Convert angle between SUMO definition and SFM definition"""
    return toDeg(expandAngle((np.pi / 2) - theta))


def limitAngle(theta):
    """Convert angle from [0,2*pi] to [-pi,pi]"""
    if isinstance(theta, np.ndarray):
        theta = np.floor(theta / (2 * np.pi)) * (-2 * np.pi) + theta

        theta[theta > np.pi] = (theta - 2 * np.pi)[theta > np.pi]
        theta[theta < -np.pi] = (theta + 2 * np.pi)[theta < -np.pi]
    else:
        theta = np.floor(theta / (2 * np.pi)) * (-2 * np.pi) + theta

        if theta > np.pi:
            theta = theta - 2 * np.pi
        elif theta < -np.pi:
            theta = theta + 2 * np.pi

    return theta


def expandAngle(theta):
    """Convert angle from [-pi,pi] [0,2*pi] to"""

    if theta < 0:
        theta = 2 * np.pi + theta

    return theta


def angleDifference(a1, a2):
    if isinstance(a1, np.ndarray):
        da = np.zeros_like(a1)

        da[a1 > a2] = (a1 - a2)[a1 > a2]
        da[a1 <= a2] = (a2 - a1)[a1 <= a2]

        da[da > np.pi] = (2 * np.pi) - da[da > np.pi]

        test_1 = np.abs(limitAngle(a1 - da) - a2)
        test_2 = np.abs(limitAngle(a1 + da) - a2)

        da[test_1 < test_2] = -da[test_1 < test_2]

        return da

    else:
        if a1 > a2:
            da = a1 - a2
        else:
            da = a2 - a1

        if da > np.pi:
            da = (2 * np.pi) - da

        test_1 = abs(limitAngle(a1 - da) - a2)
        test_2 = abs(limitAngle(a1 + da) - a2)

        if test_1 < test_2:
            return -da
        else:
            return da


def cart2polar(x, y):
    rho = np.sqrt(np.power(x, 2) + np.power(y, 2))

    phi = np.arccos(x / rho)
    if type(phi) is not np.ndarray:
        phi = np.array(phi)

    phi[y < 0] = -phi[y < 0]

    return rho, phi


def polar2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)

    return x, y


def thresh(x, minmax):
    if x < minmax[0]:
        x = minmax[0]
    elif x > minmax[1]:
        x = minmax[1]
    return x


class DiffEquation:
    """Difference Equation of a time-discrete LTI system.

    Calculates the result of an equation of the form

    y(k) = (1/a0)*(b0*u(k)+b1*u(k-1)+...+bn*u(k-n)-a1*y(k-1)-...-am*y(k-m))

    Created by Christoph Schmidt
    """

    def __init__(self, ab, y=None, u=None, th=None):
        """Initialize the difference equation.

        Parameters
        ----------

        ab : list of numpy.ndarray
            a- and b-parameters of the equation given as ([a0,a1,...,am],
                                                          [b0,b1,...,bn])
        y : numpy.ndarray, default = [0,..,0]
            Initial conditions of the system output [y(k-1), ..., y(k-m)].
            Must have length m
        u : numpy.ndarray, default = [0,..,0]
            Initial conditions of the system output [u(k-1), ..., y(k-n)].
            Must have length n
        th : float, default = No saturation.
            Symmetric output saturation.
        """

        self.a = ab[0][1:]
        self.a0inv = 1 / ab[0][0]
        self.b = ab[1]

        if y is None:
            self.y = np.zeros(len(self.a))
        else:
            self.y = y
        if u is None:
            self.u = np.zeros(len(self.b), dtype=float)
        else:
            self.u = np.zeros(len(self.b), dtype=float)
            self.u[:-1] = u

        if th is not None:
            self.th = (-th, th)
            self.y[0] = thresh(self.y[0], self.th)
            self.y[1] = thresh(self.y[1], self.th)
        else:
            self.th = None

    def __str__(self):
        """Return a string representation of the difference equation."""

        s = "y[n] = " + f"{self.b[0]:.2f}*u[n]"
        i = 1
        for b in self.b[1:]:
            b = b * self.a0inv
            s = s + f" + {b:.2f}*u[n-{i}]"
            i += 1
        i = 1
        for a in self.a[0:]:
            a = a * self.a0inv
            s = s + f" + {a:.2f}*y[n-{i}]"
            i += 1
        return s

    def step(self, uk):
        """Calculate the next time step k of the difference equation given
        the input uk

        Parameters
        ----------

        uk : float
            Input at next time step k.

        Returns
        -------

        yk : float
            Output at next time step k.

        """

        self.u = np.roll(self.u, 1)
        self.u[0] = uk

        yk = self.a0inv * (np.sum(self.b * self.u) - np.sum(self.a * self.y))

        if self.th is not None:
            yk = thresh(yk, self.th)

        self.y = np.roll(self.y, 1)
        self.y[0] = yk

        return yk

    def setInput(self, uk):
        """Add a value to the input buffer without calculating the
        difference equation.

        This may be used to keep track of input changes while an extern
        component controls the ouput dynamic instead of the difference
        equation.

        Parameters
        ----------

        yk : float
            Output at current time step k.
        """
        self.u = np.roll(self.u, 1)
        self.u[0] = uk

    def setOutput(self, yk):
        """Add a value to the output buffer without calculating the
        difference equation.

        This may be used to keep track of output changes while an extern
        component controls the ouput dynamic instead of the difference
        equation.

        Parameters
        ----------

        yk : float
            Output at current time step k.
        """
        self.y = np.roll(self.y, 1)
        self.y[0] = yk

    def update(self, ab):
        """Update the a and/or b parameters of the difference equation.

        Parameters
        ----------

        ab : list of numpy.ndarray
            a- and b-parameters of the equation given as ([a0,a1,...,am],
                                                          [b0,b1,...,bn])
        """

        if ab[0] is not None:
            self.a = ab[0][1:]
            self.a0inv = 1 / ab[0][0]
        if ab[1] is not None:
            self.b = ab[1]
