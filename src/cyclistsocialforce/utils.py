# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 16:49:49 2023.

@author: Christoph Konrad
"""

import matplotlib.pyplot as plt
import numpy as np
import io

from scipy.fft import fft

def plot_fft(t, x):
    """
    Plots the fft of a time-discrete signal x

    Parameters
    ----------
    t : array-like or float
        If an array, t is interpreted as the time samples of the datapoints in 
        x. If a float, t is interpreted as the sample time T_s
    x : array-like
        Array of N data points sampled at times t. Must be equally spaced!

    Returns
    -------
    ax : axes
        Axes of the plot.
    """
    
    n = len(x)
    
    if isinstance(t,float):
        t_s=t
        t = np.arange(0, n * t_s, t_s)
    else:
        t_s = t[1]-t[0]
        
    X = fft(x, norm='forward') #scales the output by 1/n
    
    F = np.arange(0, 1/t_s, 1/(n*t_s))
    
    #plot
    fig, ax = plt.subplots(2,1)
    ax[0].plot(t, x)
    ax[0].set_xlabel('t [s]')
    ax[1].plot(F[:int(n/2)], np.abs(X[:int(n/2)]))
    ax[1].set_xlabel('f [Hz]')
    ax[1].set_yscale('log')
    
    return ax

    
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


def to_deg(rad):
    return 360 * rad / (2 * np.pi)


def to_rad(deg):
    return 2 * np.pi * deg / (360)


def clearAxes(ax):
    for e in ax.get_children():
        e.remove()


def angleSUMOtoSFM(theta):
    """Convert angle between SUMO definition and SFM definition"""
    return limitAngle((np.pi / 2) - to_rad(theta))


def angleSFMtoSUMO(theta):
    """Convert angle between SUMO definition and SFM definition"""
    return to_deg(expandAngle((np.pi / 2) - theta))


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
    """Threshold the values in x to the limits minmax.

    x_out will be in [minmax[0], minmax[1]]

    Parameters
    ----------
    x : array-like
        Data to be thresholded.
    minmax : array-like
        Thresholds, wherew minmax[0] is the minimum threshold and minmax[1] the
        maximum.

    Returns
    -------
    x_out : array-like
        Thresholded data in [minmax[0], minmax[1]].

    """

    assert (
        minmax[0] <= minmax[1]
    ), f"Minimum must be smaller then the maximum! Instead it was [{minmax[0]}, {minmax[1]}]"
    return np.maximum(np.minimum(x, minmax[1]), minmax[0])


def validate_boolean_indicators(features, indicator_name, data_name, n_features):
    """
    Validate a boolean inicator input. This checks if features is either an
    array-like of booleans with the length n_features or an array-like of
    integers within the range [0, n_features]. 
    
    Use this for checking if a selector/indicator is compatible with the 
    data it is used to select from. 

    Parameters
    ----------
    features : array-like
        Indicator array.
    indicator_name : str
        Name of the validated boolan indicator.
        Used to create appropriate error messages.
    data_name : str
        Name of the data/varibale that this boolean selector is used on.
        Used to create appropriate error messages.
    n_features : int
        Number of selection possibilities that this selector is used on.

    Raises
    ------
    ValueError
        Raised if features doesn't satisfy the requirements to an indicator 
        array-like.

    Returns
    -------
    bool_features : np.ndarray
        The validated indicator as an array of bools with the length 
        n_features.

    """
    
    features = np.array(features)
    msg = (f"Boolean indicator array-like '{indicator_name}' must be either",
           f" an array of bool with {n_features} elements or an array of int ",
           f"in [0, {n_features}]. Instead it was an array of ",
           f"{features.dtype}.")

    if (features.dtype == bool):
        if (len(features) == n_features):
            return features
        else:
            msg = (f"Boolean indicator array-like '{indicator_name}' must ",
                   f"have the same number of elements as {data_name} has ",
                   f"features ({n_features}). Instead it was {len(features)}.")
        
    if features.dtype == int: 
        if np.all(0 < features < n_features):
            bool_features = [i in features for i in range(n_features)]
            return bool_features
        else: 
            msg = (f"Integer indices for {indicator_name} must be in [0, ",
                   f"{n_features}]. Instead it was in [",
                   f"{np.amin(features)}, {np.amax(features)}].")
             
    raise ValueError(msg)


class DiffEquation:
    """Difference Equation of a time-discrete LTI system.

    Calculates the result of an equation of the form

    y(k) = (1/a0)*(b0*u(k)+b1*u(k-1)+...+bn*u(k-n)-a1*y(k-1)-...-am*y(k-m))

    Created by Christoph Konrad
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


# -----------------------------------------------------------------------------

class Angle:
    """
    Describe planar orientations (e.g. in 2D) and angles with a complex number. 
    Concept borrowed from quaternions for 3D orientations. 
    """
    def __init__(self, complex_unitvec):
        """
        Create an Angle object.

        Parameters
        ----------
        complex_unitvec : complex
            Complex number representing an angle as cos(angle) + j sin (angle).
        """
        #assert np.abs(complex_unitvec) == 1, ("The norm of the complex angle",
        #    f"must be 1! Instead it was |{complex_unitvec}| = ",
        #    f"{np.abs(complex_unitvec)}.")
        
        self._complex_unitvec = complex_unitvec
    
    def from_euler_array(euler_array, deg=False):
        
        shape = euler_array.shape
        angle_array = np.empty(euler_array.shape, dtype=Angle).flatten()
        
        euler_flat = euler_array.flatten()
        for i in range(len(euler_flat)):
            angle_array[i] = Angle.from_euler(euler_flat[i], deg=deg)
            
        return np.reshape(angle_array, shape)
        

    def from_euler(angle, deg=False):
        """
        Create an Angle object from a given Euler angle.

        Parameters
        ----------
        angle : float
            Euler angle in rad (deg=False) or deg (deg=True).
        deg : boolean, optional
            If True, the Euler angle is interpreted in deg. 
            The default is False.

        Returns
        -------
        cyclistsocialforce.utils.Angle
            The Angle object.
        """
        
        if deg:
            angle = np.deg2rad(angle)
            return Angle(np.cos(angle) + 1j * np.sin(angle))
        else:
            return Angle(np.cos(angle) + 1j * np.sin(angle))

    def to_euler(self, deg=False):
        """
        Return the Euler angle of this Angle object. 

        Parameters
        ----------
        deg : boolean, optional
            If True, the Euler angle is expressed in deg. The default is False.

        Returns
        -------
        float
            Euler angle in rad (deg=False) or deg (deg=True).

        """
        return np.angle(self._complex_unitvec, deg=deg)
    
    def __abs__(self):
        return Angle(self._complex_unitvec.real + 1j * abs(self._complex_unitvec.imag))

    def __add__(self, other):
        assert isinstance(other, Angle)
        x = self._complex_unitvec * other._complex_unitvec
        return Angle(x)

    def __sub__(self, other):
        assert isinstance(other, Angle)
        x = self._complex_unitvec / other._complex_unitvec
        return Angle(x)

    def __mul__(self, other):
        assert isinstance(other, (int, float))
        x = self._complex_unitvec**other
        return Angle(x)

    def __div__(self, other):
        assert isinstance(other, (int, float))
        x = self._complex_unitvec ** (1 / other)
        return Angle(x)

    def __eq__(self, other):
        return self._complex_unitvec == other._complex_unitvec

    def __lt__(self, other):
        return self.to_euler() < other.to_euler()

    def __gt__(self, other):
        return self.to_euler() > other.to_euler()

    def __leq__(self, other):
        return self.to_euler() <= other.to_euler()

    def __geq__(self, other):
        return self.to_euler() >= other.to_euler()

    def __max__(self, other):
        assert isinstance(other, Angle)

        if self >= other:
            return Angle(self._complex_unitvec)
        else:
            return Angle(other._complex_unitvec)

    def __min__(self, other):
        assert isinstance(other, Angle)

        if self <= other:
            return Angle(self._complex_unitvec)
        else:
            return Angle(other._complex_unitvec)

    def __str__(self):
        return str(self.to_euler(deg=True))
        
    def __repr__(self):
        return str(self)
    
    def __float__(self):
        return self.to_euler()