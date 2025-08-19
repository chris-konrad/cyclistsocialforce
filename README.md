Cyclistsocialforce: Modified Social Forces for Cyclists with Realistic Dynamics in SUMO
==============================

This is a working repostory for a package that implements a modified social force model for cyclists. Instead of accelerations, our social forces represent the preferred velocities of a cyclist to their destination and around obstacles. This allows to introduce a various controlled models of bicycle dynamics. The original social force model for pedestrians was introduced by Helbing and Molnár (1995). Our model uses the separation into tactical and operational behavior introduced by Twaddle (2017) and currently only addresses operational behaviour. 

The model supports co-simulation with [Eclipse SUMO ](https://eclipse.dev/sumo/) via sumolib and the [TraCI](https://sumo.dlr.de/docs/TraCI.html)/[Libsumo](https://sumo.dlr.de/docs/Libsumo.html) interface.  

The model is developed for our contribution to the [2023 Bicycle and Motorcycle Dynamics Conference, 18-20 October 2023, Delft, Netherlands](https://dapp.orvium.io/deposits/649d4037c2c818c6824899bd/view), in the context of my PhD project at TU Delft. Refer to our conference proceedings preprint for more explanation. If you use this model in your research, please cite it as indicted below. 

We provide seven different bicycle models:

- `vehicle.Bicycle`: Simple two-wheeler kinematics without wheel slip. (Model from [v0.1.x](https://github.com/chrismo-schmidt/cyclistsocialforce/releases/tag/v0.1.1-bmd2023extendedabstract))

- `vehicle.InvertedPendulumBicycle`: Two-Wheeler kinematics with an inverted pendulum on top to simulate bicycle roll. A nested control loop ensures that the bicycle stays upright while following the desired yaw angle given by the social force. Additionally, the model includes new repulsive force field shapes and path planning based destination forces. Introduced with [v.1.1.0](https://github.com/chrismo-schmidt/cyclistsocialforce/releases/tag/v1.1.0-bmd2023proceedingspaper)

- `vehicle.TwoDBicycle`: Same two-wheeler kinematics as `Bicycle`, but with the modified repulsive force fields and path planning of InvertedPendulumBicycle. Introduced with [v.1.1.0](https://github.com/chrismo-schmidt/cyclistsocialforce/releases/tag/v1.1.0-bmd2023proceedingspaper)

- `vehicle.WhippleCarvalloBicycle`: Fully three-dimensional bicycle dynamics using the linearized Whipple-Carvallo model (Meijaard et al., 2007) with full-state feedback control. Inherits path-planning and repulsive forces from `vehicle.TwoDBicycle`

- `vehicle.ParticleBicycle`: A simple model of a bicycle as a mass-less particle (UNDER DEVELOPMENT).

- `vehicle.PlanarBicycle`: Simple two-wheeler kinematics without wheel slip. (UNDER DEVELOPMENT)

### Disclaimer

The package is research code under development. This is the development branch. It may contain bugs and sections of unused or insensible code as well as undocumented features. Major changes to this package are planned for the time to come. A proper API documentation is still missing. Refer to the demos and example scenarios for examples how to use this model.

## Installation

1. Install Eclipse SUMO ([instructions](https://sumo.dlr.de/docs/Installing/index.html)). Make sure that the SUMO_HOME path variable is set correctly. 

2. Clone this repository. 
   
   ```
   git clone  https://github.com/chrismo-konrad/cyclistsocialforce.git
   ```

3. Install the package and it's dependencies. Refer to `pyproject.toml` for an overview of the dependencies. 
   
   ```
   cd ./cyclistsocialforce
   pip install . 
   ```

4. A few custom dependencies are not available as pypi packages and have to be installed manually. Follow the instructions from their github pages for installation.
   
   - [`pypaperutils`](https://github.com/chris-konrad/pypaperutils) for colours of the TU Delft color scheme
   
   - [`mypyutils`](https://github.com/chris-konrad/mypyutils) for some convenience functions

## Demos

Additionally to the old demos (continue below), the package has two example scenarios in the `scenarios` that illustrate a newer, more straightforward way to build scenarios without SUMO based on a new more general `scenario` class.

- Parcours: A single cyclists following a sequence of desired destinations.

- Curve: A single cyclist following a curved road (featuring infrastructure forces)

#### Old demos

The package comes with three demos. The first demo shows a simple interaction between three cyclists in open space. The script is pre-coded with an encroachment conflict and runs as a standalone without SUMO. Running the script produces an animation of the interaction and a plot of the vehicle states.  The second demo shows co-simulation of an intersection with SUMO. It launches the SUMO GUI and creates a small scenario of a three-legged intersection with random bicycle demand. On the road segments, cyclists are controlled by SUMO. As soon as cyclists enter the intersection area, the social force model takes over control.  Movements are synchronized between SUMO and the social force model by using the TraCI interface. Switching toLibsumo is possible by uncomming a config variable in the beginning of the script, but this will [prevent simulation with the SUMO GUI](https://sumo.dlr.de/docs/Libsumo.html#limitations). A third demo simulates a larger SUMO scenario with four intersections. 

**Running the demos:**

Switch to the demo directroy.

```
cd ./demo
```

Run the standalone demo. Optionally set the `--save` flag to save a pdf of the potential and force plots to the `./demo/output/` folder. Set the `--use_inv_pendulum_bike` flag to use the `vehicle.InvertedPendulumBicycle` model instead of `vehicle.Bicycle`.

```
python demoCSFstandalone.py
```

Run the SUMO demos. After executing the line below, the SUMO GUI and matplotllib figure opens. To start the simulation, press the 'play' button in the SUMO GUI. To end it, press 'stop'. This uses the `vehicle.Bicycle` model. The inverted pendulum model currently not stable enough for crowed scenarios like this demos. 

```
python demoCSFxSUMO.py
```

Or: 

```
python demoCSFxSUMO-large.py
```

## Authors

- Christoph M. Konrad (formerly Schmidt), c.m.schmidt@tudelft.nl

License
--------------------

This package is licensed under the terms of the [MIT license](https://github.com/chrismo-schmidt/cyclistsocialforce/blob/main/LICENSE).

## Citation

If you use this model in your research, please cite it in your publications as:

Schmidt, C., Dabiri, A., Schulte, F., Happee, R. & Moore, J. (2024). Essential Bicycle Dynamics for Microscopic Traffic Simulation: An Example Using the Social Force Model [version 2; peer reviewed]. *The Evolving Scholar - BMD 2023, 5th Edition*. https://doi.org/10.59490/65a5124da90ad4aecf0ab147

## Bibliography

Helbing, D., & Molnár, P. (1995). Social force model for pedestrian dynamics. *Physical Review E*, *51*(5), 4282–4286. https://doi.org/10.1103/PhysRevE.51.4282

Hess, R., Moore, J. K., & Hubbard, M. (2012). Modeling the Manually Controlled Bicycle. *IEEE Transactions on Systems, Man, and Cybernetics - Part A: Systems and Humans*, *42*(3), 545–557. IEEE Transactions on Systems, Man, and Cybernetics - Part A: Systems and Humans. https://doi.org/10.1109/TSMCA.2011.2164244

Meijaard, J. p, Papadopoulos, J. M., Ruina, A., & Schwab, A. l. (2007). Linearized dynamics equations for the balance and steer of a bicycle: A benchmark and review. *Proceedings of the Royal Society A: Mathematical, Physical and Engineering Sciences*, *463*(2084), 1955–1982. [Linearized dynamics equations for the balance and steer of a bicycle: a benchmark and review | Proceedings of the Royal Society A: Mathematical, Physical and Engineering Sciences](https://doi.org/10.1098/rspa.2007.1857)

Twaddle, H. (2017). *Development of tactical and operational behaviour models for bicyclists based on automated video data analysis* [PhD Thesis]. Technische Universität München. https://mediatum.ub.tum.de/?id=1366878

## Project Organization

```
.
├── pyproject.toml
├── LICENSE
├── README.md
├── docs
│   └── figures
├── demo
│   ├── config
│   └── output
├── scenarios
└── src
    └── cyclistsocialforce
```
