Cyclistsocialforce: Modified Social Forces for Cyclists with Realistic Dynamics in SUMO
==============================

This is a working repostory for a package that implements a modified social force model for cyclists. Instead of accelerations, our social forces represent the preferred velocities of a cyclist to their destination and around obstacles. This allows to introduce a various controlled models of bicycle dynamics. The original social force model for pedestrians was introduced by Helbing and Molnár (1995). Our model uses the separation into tactical and operational behavior introduced by Twaddle (2017) and currently only addresses operational behaviour. 

The model supports co-simulation with [Eclipse SUMO ](https://eclipse.dev/sumo/) via sumolib and the [TraCI](https://sumo.dlr.de/docs/TraCI.html)/[Libsumo](https://sumo.dlr.de/docs/Libsumo.html) interface.  

The model is developed for our contribution to the [2023 Bicycle and Motorcycle Dynamics Conference, 18-20 October 2023, Delft, Netherlands](https://dapp.orvium.io/deposits/649d4037c2c818c6824899bd/view), in the context of my PhD project at TU Delft. Refer to our conference proceedings preprint for more explanation. If you use this model in your research, please cite it as indicted below. 

We provide three different bicycle models:

- `vehicle.Bicycle`: Simple two-wheeler kinematics without wheel slip. (Model from [v0.1.x](https://github.com/chrismo-konrad/cyclistsocialforce/releases/tag/v0.1.1-bmd2023extendedabstract))

- `vehicle.InvertedPendulumBicycle`: Two-Wheeler kinematics with an inverted pendulum on top to simulate bicycle roll. A nested control loop ensures that the bicycle stays upright while following the desired yaw angle given by the social force. Additionally, the model includes new repulsive force field shapes and path planning based destination forces. Introduced with [v.1.1.0](https://github.com/chrismo-konrad/cyclistsocialforce/releases/tag/v1.1.0-bmd2023proceedingspaper)

- `vehicle.TwoDBicycle`: Same two-wheeler kinematics as Bicycle, but with the modified repulsive force fields and path planning of InvertedPendulumBicycle. Introduced with [v.1.1.0](https://github.com/chrismo-konrad/cyclistsocialforce/releases/tag/v1.1.0-bmd2023proceedingspaper)

### Disclaimer

The package is research code under development. It may contain bugs and sections of unused or insensible code as well as undocumented features. Major changes to this package are planned for the time to come. A proper API documentation is still missing. Refer to the demos for examples how to use this model.

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

## Demos

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

- Christoph M. Konrad (formely Schmidt), c.m.schmidt@tudelft.nl

License
--------------------

This package is licensed under the terms of the [MIT license](https://github.com/chrismo-konrad/cyclistsocialforce/blob/main/LICENSE).

## Citation

If you use this model in your research, please cite it in your publications as:

Schmidt, C., Dabiri, A., Schulte, F., Happee, R. & Moore, J. (2024). Essential Bicycle Dynamics for Microscopic Traffic Simulation: An Example Using the Social Force Model  [version 2; peer reviewed]. The Evolving Scholar - BMD 2023, 5th Edition. https://doi.org/10.59490/65a5124da90ad4aecf0ab147

## Bibliography

Helbing, D., & Molnár, P. (1995). Social force model for pedestrian dynamics. *Physical Review E*, *51*(5), 4282–4286. https://doi.org/10.1103/PhysRevE.51.4282

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
└── src
    └── cyclistsocialforce
```
