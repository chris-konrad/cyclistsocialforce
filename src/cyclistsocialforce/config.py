# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 11:22:09 2023

Checks SUMO python package availability and provides global variables to
multiplex between traci and libsumo.

@author: Christoph M. Konrad
"""

# default libsumo preference. Warning: libsumo does not run with sumo-gui
_sumo_use_libsumo = False

try:
    import traci

    has_traci = True
except ImportError:
    has_traci = False

try:
    import libsumo

    has_libsumo = True
except ImportError:
    has_libsumo = False

try:
    import sumolib

    has_sumolib = True
except ImportError:
    has_sumolib = False

has_sumo = has_sumolib and (has_traci or has_libsumo)
sumo_use_libsumo = _sumo_use_libsumo and has_libsumo

if not has_sumo:
    Warning(
        (
            f"SUMO packages not found. Running without SUMO. Make sure"
            "that the sumolib and either traci or libsumo packages are"
            "installed in your environment to run with SUMO."
        )
    )
