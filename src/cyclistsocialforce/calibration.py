# -*- coding: utf-8 -*-
"""
Created on Mon May  6 16:51:47 2024

Calibration module.

Provides functionalities to calibrate different parts of the cyclist model:
    CalibrationData 
        Manage calibration data sets
    DownhillSimplexCalibration
        Fit selected parameters of the model to an objective output given
        a reference input using gradient-less downhill simplex optimisation.
        Slow but does not require to define the gradient of a simulation step.

@author: Christoph M. Schmidt
"""

from scipy.optimize import fmin
import numpy as np
import matplotlib.pyplot as plt

from trajdatamanager.datamanager import Sequence
from cyclistsocialforce.utils import validate_boolean_indicators, to_deg


### --------------------------- ERROR FUNCTIONS -------------------------------
def calc_sse_timesteps(outputs, objectives):
    """
    Calculate the Sum of Squared Errors (SSE) between all time steps of
    multiple samples with n_features features.

    Parameters
    ----------
    outputs : array-like
        List of output samples of length n_samples. Each sample must have the
        shape (n_timesamples, n_objective_features).
    objectives : array-like
        List of objective samples corresponding to the outputs. Eacht sample
        must have the (n_timesamples, n_objective_features).

    Returns
    -------
    error : array-like
        Sum of squared errors

    """
    error = 0
    for output, objective in zip(outputs, objectives):
        error += np.sum((output - objective) ** 2)
    return error


def calc_maesse_samples(outputs, objectives):
    """
    Calculate the Sum of Squared Errors (SSE) over all samples based on the
    Mean Absolute Errors (MAE) per timestep between output and objective of
    an individual sample.

    Parameters
    ----------
    outputs : array-like
        List of output samples of length n_samples. Each sample must have the
        shape (n_timesamples, n_objective_features).
    objectives : array-like
        List of objective samples corresponding to the outputs. Eacht sample
        must have the (n_timesamples, n_objective_features).

    Returns
    -------
    error : array-like
        Sum of squared errors

    """
    error = 0
    for output, objective in zip(outputs, objectives):
        error += np.mean(np.abs(output - objective)) ** 2
    return error


def objective_function_wrapper(params_vals, calibration):
    """
    General objective function wrapper for the calibration class. Needs to be
    outside of Calibration object to ensure compatibility of the
    function signature with scipy optimizers.

    Parameters
    ----------
    params_vals : array-like
        Parameter values for which the model should be evaluated.
    calibration : cyclistsocialforce.calibration.DownhillSimplexCalibration
        Calibration object.

    Returns
    -------
    error : float
        Error produced by the model when using the given parameters.

    """
    params_args = calibration._update_params_args_dict(params_vals)

    trajs, objectives = calibration.simulate_single(params_args)

    error = calibration.error_func(trajs, objectives)

    return error


### ------------------------------ CLASSES ------------------------------------


class CalibrationData(Sequence):
    """
    Maintain calibration data sets with this class.

    Bases trajdatamanager.Sequence to store collections of tracks. One
    track is a time series with multiple features (e.g. x, y, psi, v, delta,
    theta). Each track is considered one sample of the calibration dataset.

    Additionally to the basis Sequence functionality, this class provides
    methods to partition the sequence and return calibration
    reference input and objective output data when iterating over it.
    """

    def __init__(self, tracks, objective_features, input_features):
        """
        Create and input data object. This stores and manages a sequence of
        multiple tracks (time series samples), each consisting of reference
        input data and objective data.

        Parameters
        ----------
        tracks : list of trajdatamanager.datamanager.Track
            Data tracks of the calibration data set.
        objective_features : array_like of boolean
            Indicates the features of each Track that are used for the
            reference step objective. Must be an array of bool or int.
        input_features : array_like of boolean
            Indicates the features of each Track that are used for the
            reference step input. Must be an array of bool or int.
        """
        Sequence.__init__(self, tracks)

        self._validate_input(objective_features)

        self.objective_features = validate_boolean_indicators(
            objective_features, "objective_features", "tracks", tracks.shape[1]
        )
        self.input_features = validate_boolean_indicators(
            input_features, "objective_features", "tracks", tracks.shape[1]
        )

    def __iter__(self):
        """
        Iterate over the calibration data from the beginning.

        Returns
        -------
        StepCalibrationData
            The calibration data object itself.

        """
        self._i_iter = 0
        return self

    def __next__(self):
        """
        Return the next sample in an iteration of the calibration step data.

        Raises
        ------
        StopIteration
            Last sample reached.

        Returns
        -------
        s0 : np.ndarray
            Initial state s0 of this time series returned as [x, y, psi, v,
            (delta, theta)], where steer angle and roll angle are zero if not
            available in the data tracks.
        input_data : array-like
            Calibration reference input data of the nex track sample returned
            as an array of shape [n_timesteps, n_inputfeatures].
        objective_data : array-like
            Calibration objective data of the next track sample returned
            as an array of shape [n_timesteps, n_outputfeatures].

        """
        if self._i_iter < len(self.tracks):
            trk = self.tracks[self._i_iter]
            input_data = trk.data[1:, self.input_features]
            objective_data = trk.data[1:, self.objective_features]

            s0 = np.zeros(6)
            s0[0] = trk["x"][0]
            s0[1] = trk["y"][0]
            s0[2] = trk["psi"][0]
            s0[3] = trk["v"][0]

            if "delta" in trk.data_feature_keys:
                s0[4] = trk["delta"]
            if "theta" in trk.data_feature_keys:
                s0[5] = trk["theta"]

            self._i_iter += 1

            return s0, input_data, objective_data
        else:
            raise StopIteration

    def partition(self, n_seq, shares, random_seed=None):
        """
        Partition a sequence in subsequences.

        Parameters
        ----------
        n_seq : int
            Number of subsequences.
        shares : array_like
            Number of items in each subset given as a share of the total number
            of tracks in this sequence.
        random_seed : int, optional
            Random seed for random partitioning. The default is None.

        Returns
        -------
        subsequences : seq1, seq2, ..., seqn
            Randomly partitioned subsequences.
        """

        seq = Sequence.partition(self, n_seq, shares, random_seed=random_seed)

        calib_data_seq = []
        for s in seq:
            calib_data_seq.append(
                CalibrationData(
                    s.tracks, self.objective_features, self.input_features
                )
            )

        return calib_data_seq


class DownhillSimplexCalibration:
    """
    Calibrate any parameter of the cyclist model based on a gradient-less
    downhill simplex algorithm implemented by scipy.optimize.fmin (https://
    docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.fmin.html).
    This uses the Nelder-Mead algorithm (Nelder and Mead, 1965)

    Nelder, J.A. and Mead, R. (1965), “A simplex method for function
    minimization”, The Computer Journal, 7, pp. 308-313
    """

    def __init__(
        self,
        vehicle_type,
        params_keys,
        train_data,
        test_data,
        objective_features_traj,
        error_func=calc_sse_timesteps,
        fix_speed=True,
        maxiter=100,
        params_auxfuncs=None,
        params_auxfuncsargs=None,
        verbose = True
    ):
        """
        Create a DownhillSimplexCalibration object that runs the calibration
        procedure and enables testing of the resutls on sets of training and
        test scenarios given by CalibrationData objects. 

        Parameters
        ----------
        vehicle_type : type
            The type of vehicle to be calibrated. May be any
            cyclistsocialforce.vehicle.Vehicle type.
        params_keys : list
            A list of the parameters to be calibrated. The list must contain
            the exact parameter names as present Vehicle.params.
        train_data : CalibrationData
            Training data for the calibration of the model.
        test_data : CalibrationData
            Test data for the calibration of the model.
        objective_features_traj : array-like
            Indicator array for selecting the features of vehicle.traj that
            will be used to evaluate objective function.
        error_func : function, optional
            Error function for the calibration objective. The default is
            calc_sse_timesteps.
        fix_speed : Tbool, optional
            If True, the speed of the calibrated vehicle is fixed to the given
            speed in the input data. If False, the vehicle speed is modelled
            by the vehicle. The default is True.
        maxiter : int, optional
            Maximum number of iterations. The default is 100.
        params_auxfuncs : list of functions, optional
            A list of functions that transform the values returned from the
            optimizer into a parameter values of a vehicle.params field. This
            may for example be used to add the optimized the real and imaginary
            parts of the dominant poles of the vehicle dynamics the full list
            of poles that are a property of params. Each function in the list
            must correspond the respective key in params_keys:

            params_keys = ["param1", "param2"]
            params_auxfuncs = [auxfunc1, auxfunc3]

            vehicle.params.param1 = auxfunc1(optimized_params)
            vehicle.params.param2 = auxfunc2(optimized_params)

            If params_auxfuncs is None, the optimized parameters must
            directly be parameters of vehicle.params:

            vehicle.params.param1 = optimized_params[0]
            vehicle.params.param2 = optimized_params[1]

            The default is None.
        params_auxfuncsargs : list of dict, optional
            A list of dicts with additional parameters passed to the functions
            of params_auxfuncs. Must have a dict for each function in
            params_auxfuncs. Use empty dicts if a function does not take any
            additionaly parameters. May only be None if params_auxfuncs is
            None. The default is None.
        vebose : bool, optional
            Verbose output of the calibration progress / results. The default
            is true. 
        """

        self.vehicle_type = vehicle_type
        self.params_keys = params_keys

        if params_auxfuncs is not None:
            if params_auxfuncsargs is None:
                params_auxfuncsargs = [{}] * len(params_auxfuncs)
            assert len(params_auxfuncs) == len(params_keys), (
                f"If a list with auxiliary functions (params_auxfuncs) is ",
                f"provided, it has to have the same number of elements as ",
                f"'params_keys' ({len(params_keys)}). Instead it had",
                f"{len(params_auxfuncs)} elements.",
            )
            assert len(params_auxfuncsargs) == len(params_keys), (
                f"If a list with auxiliary function parameters (params_",
                f"auxfuncs) is provided, it has to have the same number of ",
                f"elements as 'params_keys' {len(params_keys)}. Instead it ",
                f"had {len(params_auxfuncsargs)} elements.",
            )
        self.params_auxfuncs = params_auxfuncs
        self.params_auxfuncsargs = params_auxfuncsargs

        self.train_data = train_data
        self.test_data = test_data
        self.objective_features_traj = validate_boolean_indicators(
            objective_features_traj,
            "objective_features_traj",
            "vehicle.traj",
            6,
        )
        self.fix_speed = fix_speed
        self.error_func = error_func
        self.maxiter = maxiter
        self.param_args_opt = None
        self.verbose = verbose

    def _update_params_args_dict(self, params_vals):
        """
        Update the parameter arguments dictionary with the current parameter
        values.

        Parameters
        ----------
        params_vals : array-like
            Current parameter values. Must either correspond directly to 
            params_keys or after transformation through params_auxfuncs.

        Returns
        -------
        params_args : dict
            Parameter arguments dictionary.

        """

        params_args = {}

        if self.params_auxfuncs is None:
            for key, val in zip(self.params_keys, params_vals):
                params_args[key] = val
        else:
            for key, auxfunc, auxfuncargs in zip(
                self.params_keys,
                self.params_auxfuncs,
                self.params_auxfuncsargs,
            ):
                params_args[key] = auxfunc(params_vals, **auxfuncargs)

        return params_args

    def simulate_single(self, params_args, return_vehicles=False, test=False):
        """
        Simulate a single vehicle for the current parameter set params_args and
        all scenario samples from train_data (test = False) or test_data 
        (test = True).

        Parameters
        ----------
        params_args : dict
            Parameter dict like params_args[params_key] = params_val with key-
            value pairs corresponding to parameters of vehicle.params.
        return_vehicles : TYPE, optional
            If true, returns the vehicle instances for all samples after the simulation
            finished. If False, only the objective features of vehicle.traj
            are returned. The default is False.
        test : bool, optional
            If True, simulates the test scenarios. If False, simulates the 
            training scenarios. The default is False.

        Returns
        -------
        results : list
            List of results for all scenario samples. If return_vehicles, 
            this contains the vehicle instances after simultion. If not
            return_vehicles, this only contains the objective features from 
            each vehicle.traj
        objectives : list
            List of referece objective outputs for all scenario samples.

        """

        params = self.vehicle_type.PARAMS_TYPE(**params_args)

        results = []
        objectives = []

        if test:
            data = self.test_data
        else:
            data = self.train_data

        for s0, input_data, objective_data in data:

            objectives.append(objective_data)

            # create vehicle
            vehicle = self.vehicle_type(
                s0,
                vid=self.vehicle_type.__name__,
                params=params,
                saveForces=True,
            )

            # run simulation
            n = input_data.shape[0]

            for i in range(n):

                if self.fix_speed:
                    vehicle.s[3] = np.sqrt(
                        input_data[i, 0] ** 2 + input_data[i, 1] ** 2
                    )

                vehicle.step(input_data[i, 0], input_data[i, 1])

            # pack output
            if return_vehicles:
                results.append(vehicle)
            else:
                results.append(
                    vehicle.traj[self.objective_features_traj, :n].T
                )

        return results, objectives

    def run(self, params_vals_guess):
        """
        Run the calibration.

        Parameters
        ----------
        params_vals_guess : array-like
            Initial guess of the calibration parameters. The elements of
            params_vals_guess must either correspond directly to the keys
            in params_keys:
                vehicle.params.key0 = params_vals_guess[0]
                ...
                vehicle.params.keyn = params_vals_guess[n]
                
            or to the signature of the params_auxfuncs functions:
                vehicle.params.key0 = params_auxfunc0(params_vals_guess)
                ...
                vehicle.params.keyn = params_auxfuncn(params_vals_guess)

        Returns
        -------
        results : list
            Results, where results[0] is a list of the results returned by
            scipy.optimize.fmin and results[1] are the vehicle instances after
            simulation with the optimal parameters.

        """
        if self.verbose: print(f"Calibrating {self.vehicle_type.__name__} ...")

        # run optimization
        results = fmin(
            objective_function_wrapper,
            params_vals_guess,
            (self,),
            full_output=True,
            maxiter=self.maxiter,
        )

        # print results
        param_args = self._update_params_args_dict(results[0])
        if self.verbose:
            for key in param_args.keys():
                print(f"         {key}: {param_args[key]}")
        
        # simulate with optimal parameters and pack results
        results = list(results)
        results.append(
            self.simulate_single(
                self._update_params_args_dict(results[0]), return_vehicles=True
            )
        )

        self.param_args_opt = param_args

        return results

    def test(
        self,
        param_args_opt=None,
        plot_results=False,
        color="blue",
        axes=None,
        name=None,
        plot_inref=True,
    ):
        """
        Test the optimization results on the test scenarios and plot results.

        Parameters
        ----------
        param_args_opt : dict, optional
            Parameter argument dictionary with the parameters to be tests. If
            None, the optimal parameters determined by 
            DownhillSimplexCalibration.run() are used. The default is None.
        plot_results : bool, optional
            If True the results are plotted. The default is False.
        color : color, optional
            Matplotlib color definition for the color of the line plot of the
            simulation result. The default is "blue".
        axes : List of Axes, optional
            List of axes to be plot in. The list has to have the same number
            of elements as there are test scenarios in the test dataset. If 
            None, a new set of axes in a single figure is created. 
            The default is None.
        name : str, optional
            A label for the line plot of this result. The default is None.
        plot_inref : bool, optional
            If true, adds the reference input to the plot. The default is True.

        Returns
        -------
        error : float
            The error of the calibrated model when tasked with the test 
            scenarios.
        vehicles : list
            List of vehicles after simulation of the test scenarios.

        """

        if self.verbose: 
            print((f"Testing calibration against {len(self.test_data.tracks)}",
                   f" test samples ..."))

        # run simulation with optimal parameters
        if param_args_opt is None:
            assert self.param_args_opt is not None, (
                "First run the calibration",
                "using run() before testing.",
            )

            param_args_opt = self.param_args_opt

        vehicles, objectives = self.simulate_single(
            param_args_opt, test=True, return_vehicles=True
        )

        # extract results
        trajs = [
            veh.traj[self.objective_features_traj, 1 : veh.i + 1].T
            for veh in vehicles
        ]
        error = self.error_func(trajs, objectives)

        # plot 
        if plot_results:
            n = len(self.test_data.tracks)
            if axes is None:
                fig, axes = plt.subplots(1, n, sharey=True)
            for (s0, input_data, objective_data), traj, ax in zip(
                self.test_data, trajs, axes
            ):
                if plot_inref:
                    ax.plot(
                        to_deg(objective_data - s0[2]),
                        color="gray",
                        label="measurement",
                    )
                    ax.plot(
                        to_deg(
                            np.arctan(input_data[:, 1] / input_data[:, 0])
                            - s0[2]
                        ),
                        color="gray",
                        linestyle="--",
                        label="reference input",
                    )
                ax.plot(to_deg(traj - s0[2]), color=color, label=name)
        
        # print results to console
        if self.verbose: print(f"    SSE: {error:.4f}")

        return error, vehicles
