""" Visual verification of the pole models for the BalancingRiderBicycle.

Creates a plot of the model poles vs. speed for the default mode (component means)
and stochastic sampling. 

This also illustrates that the pole locations change independently from the previous 
pole with stochastic sampling. 
"""
import argparse
import importlib.resources as resources

from cyclistsocialforce.controlbehavior import PoleModel
from cyclistsocialforce.vehicle import BalancingRiderBicycle
from cyclistsocialforce.parameters import BalancingRiderBicycleParameters
import numpy as np
import matplotlib.pyplot as plt


def main():
    
    s0 = [0,0,0,5,0,0,0,0]
    speeds = np.linspace(1.5, 5.5, 30)
    fig, axes = plt.subplots(1,5, sharex=True, layout='constrained')

    # Mean pole locations. Should equal figure 18 in https://engrxiv.org/preprint/view/6107
    for c in range(0,2):
        poles = []
        params = BalancingRiderBicycleParameters(controlparam_polemodel_component=c)
        bike = BalancingRiderBicycle(s0, params=params)
        for v in speeds:
            bike.params.update_control_params(v)
            print(bike.params.poles)
            poles.append(bike.params.poles)

        poles = np.array(poles)
    
        axes[0].plot(speeds, poles[:,0])
        axes[0].set_ylabel("sigma_0")
        axes[1].plot(speeds, np.real(poles[:,1]))
        axes[1].set_ylabel("sigma_1")
        axes[2].plot(speeds, -np.imag(poles[:,2]))
        axes[2].set_ylabel("omega_1")
        axes[3].plot(speeds, np.real(poles[:,3]))
        axes[3].set_ylabel("sigma_2")
        axes[4].plot(speeds, -np.imag(poles[:,4]), label=f"mean poles component {c}")
        axes[4].set_ylabel("omega_2")
    
    # Pole resampling for stochastic rider behavior
    poles = []
    params = BalancingRiderBicycleParameters(stochastic_control_behavior=True)
    bike = BalancingRiderBicycle(s0, params=params)
    for v in speeds:
        bike.params.update_control_params(v)
        print(bike.params.poles)
        poles.append(bike.params.poles)

    poles = np.array(poles)

    axes[0].plot(speeds, poles[:,0])
    axes[0].set_ylabel("sigma_0")
    axes[1].plot(speeds, np.real(poles[:,1]))
    axes[1].set_ylabel("sigma_1")
    axes[2].plot(speeds, -np.imag(poles[:,2]))
    axes[2].set_ylabel("omega_1")
    axes[3].plot(speeds, np.real(poles[:,3]))
    axes[3].set_ylabel("sigma_2")
    axes[4].plot(speeds, -np.imag(poles[:,4]), label=f"stochastic behavior")
    axes[4].set_ylabel("omega_2")
    axes[4].legend()

    plt.suptitle("Balancing Rider Behavior Model - pole locations")
    plt.show(block=True)
    

if __name__ == "__main__":
    main()

