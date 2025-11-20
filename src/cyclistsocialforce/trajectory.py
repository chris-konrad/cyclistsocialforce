# -*- coding: utf-8 -*-
"""
Created on Fri Apr 21 10:52:45 2023.

@author: Christoph Konrad
"""
from numpy import linspace
from scipy import interpolate


def generateSplinePrototype(x, y, npoints=5):
    """
    Generate a cubic spline trajectory prototype based on four given points.

    Parameters
    ----------
    x : array
        Array of x locations. x and y must be same length (at least 4).
    y : array
        Array of y locations. x and y must be same length (at least 4).
    npoints : int (default=5)
        Number of points in the trajectory prototype.

    Returns
    -------
    x_p : array
        Array of x locations of the trajectory prototype.
    y_p : array
        Array of y locations of the trajectory prototype.

    """

    assert len(x) == len(y), "x and y must be same length!"
    assert (
        len(x) >= 3
    ), "Provide at least 3 points to calculate a cubic trajectory prototype"

    tck, u = interpolate.splprep((x, y), s=0.0)
    x_p, y_p = interpolate.splev(linspace(0, 1, npoints), tck)

    return x_p, y_p
