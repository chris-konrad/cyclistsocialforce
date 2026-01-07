# -*- coding: utf-8 -*-
"""
Model pole distributions representing behavioral parameters and test the distribution of their trajectory predictions.

@author: Christoph M. Konrad
"""

import os
# Prevent memory leakage of KNN on Windows (only use for pole model calibration)
#if os.name == 'nt':
#    os.environ["OMP_NUM_THREADS"] = "1"

import re
import yaml

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from datetime import datetime

from sklearn.mixture import GaussianMixture as SklearnGaussianMixture
from sklearn.mixture._gaussian_mixture import _compute_precision_cholesky
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.model_selection import cross_validate
from sklearn.linear_model import LinearRegression

from scipy.stats import multivariate_normal
from mypyutils.io import read_yaml

from pypaperutils.design import TUDcolors

tudcolors = TUDcolors()
cmap = tudcolors.colormap()

#global constants
T_S = 0.01

def get_outliers_all_models(paths, models):
    """ Combine outliers across models. 
    """
    
    outlier_cols = []
    for i, m in enumerate(models):
        filepath_poles = paths.getfilepath_pm_sortedpoles(m)

        df_i = pd.read_csv(filepath_poles, sep=";")[['sample_id', 'outliers']]
        col = f'outliers_{m}'
        outlier_cols.append(col)

        if i == 0:
            df = df_i.rename(columns={'outliers': col})
        else:
            df = df.merge(df_i, on='sample_id')
            df.rename(columns={'outliers': col}, inplace=True)
            
    df['outliers'] = df[outlier_cols].any(axis=1)

    print(f"Number of samples: {df.shape[0]-df['outliers'].sum()} (ignoring {df['outliers'].sum()} outliers across {models})")

    return df

def polefeaturetable_to_polearray(polefeature_table, features='ImRe'):
    """ Convert pole features in a pandas table into an array of complex-valued poles
    """

    poles = []

    if features == 'ImRe':
        for i in range(10):
            key_real = f"p{i}_real"
            key_imag = f"p{i}_imag"

            p_i = np.zeros(polefeature_table.shape[0], dtype=complex)

            if key_real in polefeature_table:
                p_i += polefeature_table[key_real].to_numpy().flatten()
            if key_imag in polefeature_table:
                p_i += 1j * polefeature_table[key_imag].to_numpy().flatten()
            
            if (key_imag not in polefeature_table) and (key_real not in polefeature_table):
                break

            poles.append(p_i)

            if np.any(np.imag(p_i) != 0.0):
                poles.append(np.conjugate(p_i))

    elif features == 'AngMag':
        for i in range(10):
            key_real = f"p{i}_real"
            key_ang = f"p{i}_ang"
            key_mag = f"p{i}_mag"

            p_i = np.zeros(polefeature_table.shape[0], dtype=complex)

            if key_real in polefeature_table:
                p_i += polefeature_table[key_real].to_numpy().flatten()
            elif key_ang in polefeature_table and key_mag in polefeature_table:
                p_i += polefeature_table[key_mag] * (np.cos(polefeature_table[key_ang]) + 1j * np.sin(polefeature_table[key_ang]))
            else:
                break

            poles.append(p_i)

            if np.any(np.imag(p_i) != 0.0):
                poles.append(np.conjugate(p_i))
            
    poles = np.array(poles).T

    return poles


def score_gmm(gmm, X):
    """ Compute the multimetric score of a gaussian mixture model.
    """
    if type(gmm) != GaussianMixture:
        raise ValueError(f"'gmm' must be sklearn.mixture.GaussianMixture. Instead it was {type(gmm)}")

    score = {'BIC': gmm.bic(X),
            'AIC': gmm.aic(X),
            'NLL': -gmm.score(X)}
    return score
    

def score_conditional_gmm(gmm, X):
    """ Compute the multimetric score of a conditional gaussian mixture model.
    """
    
    if type(gmm) != ConditionalGaussianMixture:
        raise ValueError(f"'gmm' must be ConditionalGaussianMixture. Instead it was {type(gmm)}")

    scores = []

    feature_index_rest = [n for n in range(X.shape[1]) if n != gmm.feature_index_given]

    for i in range(X.shape[0]):
            
        X_given = X[i, gmm.feature_index_given]
        X_i = X[i, feature_index_rest].reshape(1, len(feature_index_rest))

        gmm_cond = gmm._get_conditional_gmm(X_given)

        scores.append([gmm_cond.bic(X_i), gmm_cond.aic(X_i), -gmm_cond.score(X_i)])

    scores = np.array(scores)
    scores = np.mean(scores, axis=0)

    return {'BIC': scores[0],
            'AIC': scores[1],
            'NLL': scores[2]}



class GaussianMixture(SklearnGaussianMixture):
    """ A class extending sklearn's Gaussian Mixture with functionality to create objects from known parameters,
    and custom pdf evaluation. Additionally enables to scale the variance of all components.
    """

    def __init__(self, n_components=1, n_init=100, covariance_type='full', variance_scale=1.0, **kwargs):
        """ Create a GaussianMixture object.

        Parameters
        ----------

        n_components : int, optional
            Number of Gaussian components. Default is 1.
        n_init : int, optional
            Number of initializations for fitting. Defualt is 100.
        covariance_type : str, optional
            The covariance type. Default is "full".
        variance_scale : float, optional
            Variance scale factor applied to all components. Default is 1.0.
        kwargs : dict
            All other keyword arguments from Sklearn's GaussianMixture.
        """
        self.variance_scale=variance_scale
        super().__init__(n_components=n_components, covariance_type=covariance_type, n_init=n_init, **kwargs)


    def from_parameters(means, covariances, weights, **kwargs):
        """ Create a multivariate Gaussian Mixture model from known / converged parameters.

        Parameters
        ----------
        means : array-like
            Array of means. Must be shaped (n_features, n_components).
        covariances : array-like
            Array of covariances. Must be shaped (n_components, n_features, n_features).
        weights : array-like
            Array of component weights. Must be size n_components. Weights must sum to 1.0
        kwargs : dict
            Any OTHER keyword arguments for sklearn.mixture.GaussianMixture

        Returns
        -------
        gmm : GaussianMixture
            A GaussianMixture object with the given parameters. 

        """

        means = np.array(means)
        covariances = np.array(covariances)

        n_features = means.shape[1]
        n_components = means.shape[0]

        if not np.all(covariances.shape == np.array((n_components, n_features, n_features))):
            msg = (f"n_features={n_features} and n_components={n_components} inferred from means.shape. "
                   f"'covariances' must be shaped [{n_components},{n_features},{n_features}]. Instead it "
                   f"was {covariances.shape}")
            raise ValueError(msg)
        
        weights = np.array(weights).flatten()
        if weights.size != n_components:
            msg = (f"n_components={n_components} inferred from means.shape. "
                   f"'weights' must be size {n_components}. Instead it "
                   f"was {weights.size}")
        if np.sum(weights) != 1.0:
            msg = (f"Weights do not sum to one!")
        
        gmm = GaussianMixture(n_components=n_components, covariance_type='full', **kwargs)
        gmm.means_ = means
        gmm.covariances_ = covariances
        gmm.weights_ = weights
        gmm.precisions_cholesky_ = _compute_precision_cholesky(covariances, 'full')
        gmm.converged_ = True
        gmm.n_iter_ = 1

        return gmm
    
    def fit(self, X):
        """ Fit the Gaussian Mixture model to the given Data.

        Parameters
        ----------
        X : array-like
            Data matrix.
        """

        super().fit(X)

        if self.variance_scale != 1.0:
            cov = self.get_full_covariancematrix()
            S = np.eye(cov.shape[1]) * np.sqrt(self.variance_scale)
            for i in range(cov.shape[0]):
                cov[i,:,:] = S @ cov[i,:,:] @ S.T

            self.covariances_ = cov
            self.covariance_type = 'full'
            self.precisions_cholesky_ = _compute_precision_cholesky(cov, 'full')

        return self

    def get_full_covariancematrix(self):
        """ Return the full covariance matrix of the fitted Gaussian Mixture model, 
        even if the covariance type is tied/diag or spherical.

        Returns
        -------
        covariances : array like
            Covariance matrices shaped [n_features, n_features, n-components]
        """
        n_features = self.means_.shape[1]

        if self.covariance_type == 'full':
            return self.covariances_
        elif self.covariance_type == "tied":
            return np.tile(self.covariances_[np.newaxis, :, :], (self.n_components,1,1))
        elif self.covariance_type == "diag":
            return np.array([np.diag(self.covariances_[k]) for k in range(self.n_components)])
        elif self.covariance_type == 'spherical':
            return np.array([np.eye(n_features) * self.covariances_[k] for k in range(self.n_components)])
        raise RuntimeError(f"Illegal covariance type {self.covariance_type}!")
    
    def eval_1d_marginal_pdf_samples(self, samples, idx_x):
        """ Evaluate the marignal pdf of a selected feature x at sample locations.

        Parameters
        ----------
        samples : array_like
            Sample values of feature x to evaluate the marinal pdf at.
        idx_x : int
            The index of feature x. 

        Returns
        -------
        samples : array_like
            Samples of feature x (same as input)
        densities : array_like
            Marginal densities of feature x at the sample locations
        """

        # accumulate densities
        densities = np.zeros_like(samples)
        cov = self.get_full_covariancematrix()
        
        for k in range(self.n_components):
            mean_k = self.means_[k, idx_x]
            var_k = cov[k][idx_x, idx_x]

            densities_k = self.weights_[k] * multivariate_normal(mean=mean_k, cov=var_k).pdf(samples)
            densities += densities_k

        return samples.flatten(), densities.flatten()

    def eval_1d_marginal_pdf(self, xlim, idx_x, n_samples=200):
        """ Evaluate the marignal pdf of a selected feature x across a range.

        Parameters
        ----------
        xlim : tuple
            A tuple specifying the range to evaluate as [min, max].
        idx_x : int
            The index of feature x. 
        n_samples : int, optional
            Number of samples with the given limits. Default is 200.

        Returns
        -------
        samples : array_like
            Samples of feature x (same as input)
        densities : array_like
            Marginal densities of feature x at the sample locations
        """
        
        # grid
        locations = np.linspace(xlim[0], xlim[1], n_samples)

        return self.eval_1d_marginal_pdf_samples(locations, idx_x)

    def eval_2d_marginal_pdf(self, xlim, ylim, idx_x, idx_y, n_samples=200):
        """ Evaluate the 2d marignal pdf of a pair of features x and y across a range.

        Parameters
        ----------
        xlim : tuple
            A tuple specifying the range of feature x to evaluate as [min, max].
        ylim : tuple
            A tuple specifying the range of feature y to evaluate as [min, max].
        idx_x : int
            The index of feature x. 
        n_samples : int, optional
            Number of samples with the given limits. Default is 200.

        Returns
        -------
        samples : array_like
            Samples of feature x (same as input)
        densities : array_like
            Marginal densities of feature x at the sample locations
        """

        # grid
        x = np.linspace(xlim[0], xlim[1], n_samples)
        y = np.linspace(ylim[0], ylim[1], n_samples)
        X, Y = np.meshgrid(x, y)
        locations = np.dstack((X, Y))

        # accumulate densities
        densities = np.zeros_like(X)
        cov = self.get_full_covariancematrix()
        

        for k in range(self.n_components):
            mean_k = [self.means_[k, idx_x], self.means_[k, idx_y]]
                
            cov_k = cov[k]
            cov_k = cov_k[[idx_x, idx_y],:][:, [idx_x, idx_y]]

            densities_k = self.weights_[k] * multivariate_normal(mean=mean_k, cov=cov_k).pdf(locations)
            densities += densities_k

        return locations.reshape(-1,2), densities.flatten()


class ConditionalGaussianMixture(GaussianMixture):
    """ Describes a conditional multivariate GaussianMixture.
    """

    def __init__(self, feature_index_given=1, n_components=1, n_init=100, covariance_type='full', **kwargs):
        """ Create a ConditionalGaussianMixture object.

        Parameters
        ----------
        feature_index_given : int, optional
            The index of the conditional feature in the data matrix. 
        n_components : int, optional
            Number of Gaussian components. Default is 1.
        n_init : int, optional
            Number of initializations for fitting. Defualt is 100.
        covariance_type : str, optional
            The covariance type. Default is "full".
        variance_scale : float, optional
            Variance scale factor applied to all components. Default is 1.0.
        kwargs : dict
            All other keyword arguments from Sklearn's GaussianMixture and this modules' GaussianMixture.
        """

        super().__init__(n_components=n_components, covariance_type=covariance_type, n_init=n_init, **kwargs)

        self.feature_index_given = feature_index_given

        #if not self.covariance_type == 'full':
        #    raise ValueError("Covariance type must be 'full'!")

    
    def from_parameters(means, covariances, weights, feature_index_given, **kwargs):
        """ Create a multivariate Gaussian Mixture model from known / converged parameters.

        Parameters
        ----------
        means : array-like
            Array of means. Must be shaped (n_features, n_components).
        covariances : array-like
            Array of covariances. Must be shaped (n_components, n_features, n_features).
        weights : array-like
            Array of component weights. Must be size n_components. Weights must sum to 1.0
        feature_index_given : int
            Index of the given feature.
        kwargs : dict
            Any OTHER keyword arguments for sklearn.mixture.GaussianMixture

        Returns
        -------
        gmm : GaussianMixture
            A GaussianMixture object with the given parameters. 

        """

        means = np.array(means)
        covariances = np.array(covariances)

        n_features = means.shape[1]
        n_components = means.shape[0]

        if not np.all(covariances.shape == np.array((n_components, n_features, n_features))):
            msg = (f"n_features={n_features} and n_components={n_components} inferred from means.shape. "
                   f"'covariances' must be shaped [{n_components},{n_features},{n_features}]. Instead it "
                   f"was {covariances.shape}")
            raise ValueError(msg)
        
        weights = np.array(weights).flatten()
        if weights.size != n_components:
            msg = (f"n_components={n_components} inferred from means.shape. "
                   f"'weights' must be size {n_components}. Instead it "
                   f"was {weights.size}")
        if np.sum(weights) != 1.0:
            msg = (f"Weights do not sum to one!")
        
        gmm = ConditionalGaussianMixture(feature_index_given=feature_index_given, n_components=n_components, covariance_type='full', **kwargs)
        gmm.means_ = means
        gmm.covariances_ = covariances
        gmm.weights_ = weights
        gmm.precisions_cholesky_ = _compute_precision_cholesky(covariances, 'full')
        gmm.converged_ = True
        gmm.n_iter_ = 1

        return gmm


    def fit(self, X):
        """ Fit the conditional Gaussian Mixture model to the given data.
        Should include samples for the feature to be conditioned on. 

        Parameters
        ----------
        X : array-like
            Data matrix.
        """
        self.feature_indices_marginals = [i for i in np.arange(X.shape[1]) if i != self.feature_index_given]
        super().fit(X)
        return self


    def _get_conditional_gmm(self, x_given):
        """ Return a GaussianMixture object modeling the distribution conditioned on X_given"""

        cov = self.get_full_covariancematrix()
        mu = self.means_
        pi = self.weights_

        idx_given = np.array(self.feature_index_given)
        idx_cond = [n for n in range(self.means_[0].size) if n not in idx_given]

        n_features = self.means_.shape[1]
        n_given = idx_given.size
        n_cond = n_features - n_given

        x_given = np.reshape(x_given, (n_given, 1))
        
        # make masks to form the conditional covariance matrices
        # given: the given feature to be conditioned on
        # cond: the remaining conditional distribution
        mask_given = np.zeros((n_features, n_features), dtype=bool)
        mask_given[idx_given, :] = True
        mask_given[:, idx_given] = True

        mask_cond_cov = np.logical_not(mask_given)

        cov_cond = []
        mu_cond = []
        pi_cond = []

        for n in range(self.n_components):
            cov_n = cov[n,:,:]
            mu_n = mu[n,:]

            var_given_n = cov_n[idx_given, idx_given].reshape(n_given,n_given)
            cov_given_n = cov_n[idx_cond, idx_given].reshape(n_cond,n_given)
            mu_given_n = mu_n[idx_given].reshape(n_given, 1)

            mu_cond_n = mu_n[idx_cond].reshape(n_cond, 1) + (cov_given_n @ np.linalg.inv(var_given_n)) @ (x_given - mu_given_n)
            cov_cond_n = cov_n[mask_cond_cov].reshape(n_cond, n_cond) - (cov_given_n @ np.linalg.inv(var_given_n) @ cov_given_n.T)

            pi_cond_n = pi[n] * multivariate_normal.pdf(x_given, mu_given_n, var_given_n)

            cov_cond.append(cov_cond_n)
            mu_cond.append(mu_cond_n.flatten())
            pi_cond.append(pi_cond_n)

        pi_cond = np.array(pi_cond) / np.sum(pi_cond)
        if np.any(pi_cond==0.0):
            #prevent pi_cond from getting 0 to suppress warnings later on
            pi_cond[pi_cond == 0.0] = np.finfo(float).eps * self.n_components
            pi_cond = pi_cond/np.sum(pi_cond)
        cov_cond = np.array(cov_cond)

        gmm_cond = GaussianMixture.from_parameters(mu_cond, cov_cond, pi_cond, random_state=self.random_state)

        return gmm_cond 
    
    
    def sample(self, n_samples=1, X_given=[0.0]):
        """ Draw samples from the conditional distribution. Draws n_samples per given feature value X_given to be conditioned on.

        Parameters
        ----------
        n_samples : int, optional
            Number of samples to draw. Default is 1.
        X_given : list-like, optional
            List of n_given feature values to be conditioned on. Default is [0.0].
        
        Returns
        -------
        samples : np.ndarray
            Array of drawn samples shape (n_given, n_samples, n_features). If n_samples==1, the shape is (n_samples, n_features)
        """ 

        if not isinstance(X_given, (list, tuple, np.ndarray)):
            X_given = list(X_given)
        
        samples = []
        labels = []
        for x_given in X_given:
            gmm_cond = self._get_conditional_gmm(x_given)
            samples_i, labels_i = gmm_cond.sample(n_samples=n_samples)
            samples.append(samples_i)
            labels.append(labels_i)

        if len(X_given)>1:
            samples = np.array(samples)
        else:
            samples = samples_i

        labels = np.array(labels).flatten()

        return samples, labels
    

    def eval_conditional_marginal_pdf(self, ylim, x_given, idx_y, n_samples=200):
        """ Evaluate the marginal conditional pdf N(Y=y|X=x_given). 

        Parameters
        ----------

        ylim : list
            Range of y
        x_given : float
            The given value to be conditioned on
        idx_y : int
            The id of the requested marginal. May not be the ID of the conditional. 
        n_samples : int
            Number of samples for the marginal distribution between ylim[0] and ylim[1]
        """

        if idx_y == self.feature_index_given:
            raise ValueError("The requested marginal can't be the one that is conditoned on!")
        
        #convert to index of the conditional distribution
        idx_y = self.feature_indices_marginals.index(idx_y)
        
        #conditional gmm
        gmm_cond = self._get_conditional_gmm(x_given)

        #range
        y = np.linspace(ylim[0], ylim[1], n_samples)
        densities = np.zeros_like(y)
    
        #accumulate densities
        for k in range(self.n_components):

            mean_k = gmm_cond.means_[k][idx_y]
            cov_k = gmm_cond.covariances_[k][idx_y, idx_y]

            densities += gmm_cond.weights_[k] * multivariate_normal.pdf(y, mean=mean_k, cov=cov_k)
        
        return y, densities

    
class LogTransformer():
    """ A data transformer applying a log-shift transformation to the given data:
    
    y = log(x-1)

    """

    def __init__(self, alpha=0.9):
        """ Create a logshift transformer object.

        Parameters
        ----------
        alpha : float, optional
            Factor for fitting the shift parameter. a will be chosen as alpha * min(X).
            Must be in [0,1]. Default is 0.9
        """
        self.alpha = alpha
        if alpha <= 0.0 or alpha >= 1.0:
            raise ValueError(f"alpha must be in ]0,1[. Instead it was {alpha}!")

        self.a_ = None  
        self.sign_ = None


    def _check_X(self, X, limit=None):
        if limit is None:
            limit = self.a_
        if np.any((X - limit) <= 0):
            raise ValueError(f"All elements of X must be larger then {limit}!")


    def fit(self, X, y=None):
        """ Fit the logshift transformer.

        Determines:
            - required sign to enable log
            - a from the smallest value in (postitive) X

        Return
        ------
        LogTransformer
            The fitted logshift transformer. 
        """

        X = np.asarray(X)

        # find sign 
        self.sign_ = np.sign(X[0,:]).reshape((1,-1))
        X = X * self.sign_

        self._check_X(X, limit=0)
        
        # find shift parameter
        self.a_ = self.alpha * np.min(X, axis=0)
        self.a_ = self.a_.reshape(1, X.shape[1])

        return self


    def transform(self, X):
        """ Apply the logshift trafo to the data in X.
        """

        X = np.asarray(X)
        
        X = X * self.sign_
        self._check_X(X)
        if self.a_ is None:
            raise RuntimeError("The transformer has not been fitted yet.")
        return np.log(X - self.a_)
    
    def inverse_transform(self, X):
        """ Apply the inverse logshift trafo to the data in X.
        """

        X = np.asarray(X)

        if self.a_ is None:
            raise RuntimeError("The transformer has not been fitted yet.")
        Y = np.exp(X) + self.a_
        Y = Y * self.sign_

        return Y


class PreprocessingPipeline():
    """ A pipeline applying multiple preprocessing steps after another."""

    POWER_TRANSFORMS = ("yeo-johnson", "box-cox", "none")

    def __init__(self, feature_set, features, normalize=True, log_transform=True, power_transform="yeo-johnson", save=False, dir_out=None, tag=None):
        """ Create a Preprocessing pipeline object. 
        
        Parameters
        ----------
        feature_set : str
            Name of any of the feature sets (See PoleModel).
        features : list
            The features of the feature set (see PoleModel).
        normalize : bool, optional
            Normalize the data. Default is True
        log_transform : bool, optional
            Apply log transform. Default is True
        power_transform : str, optional
            Apply a power transformation. Can be "yeo-johnson", "box-cox", "none". 
            Default is "yeo-johnson"
        save : bool, optional
            If True, automatically saves a plot form fitting. 
        dir_out : str, optional
            The directory for saving the plot.
        tag : str, optional
            A tag for the name of the figure.
        """
        
        #Transforms
        self.normalize = normalize

        if power_transform in self.POWER_TRANSFORMS:
            self.power_transform = power_transform
        else:
            raise NotImplementedError(f"Power transformation '{power_transform}' not implemented! Choose any of {self.POWER_TRANSFORMS}.")
    
        self.log_transform = log_transform

        #
        self.transformers_ = []
        self.is_fitted_ = False

        #Output
        self.feature_set = feature_set
        self.features = features
        self.save = save
        self.dir_out = dir_out
        self.tag = tag

        method_str = ""
        if power_transform != 'none':
            method_str+=f"{power_transform}-pt, "
        if normalize:
            method_str+=f"normalized, "
        method_str = method_str[:-2]  

        self.method_str = method_str

    def from_parameters(feature_set, features, normalize=False, power_transform="yeo-johnson", log_transform=False, 
                        power_transform_params={}, standard_scaler_params={}, log_transform_params={}, 
                        save=False, dir_out=None, tag=None):
        """ Create a ProcessingPipeline object from known parameters.
        """
        if power_transform !="yeo-johnson":
            raise NotImplementedError("Initializing a PreprocessingPipeline with other power transformation then yeo-johnson is not implemented!")

        pipe = PreprocessingPipeline(feature_set, features, normalize=normalize, power_transform=power_transform, 
                                     log_transform=log_transform, save=save, dir_out=dir_out, tag=tag)
        
        pipe.method_list = [f'{pipe.power_transform}-power-transform']
        pipe.n_features = len(pipe.features)

        if log_transform:
            pipe.log_transform_features_ = np.array(log_transform_params["log_transform_features"])
            pipe.transformers_.append(LogTransformer())
            pipe.transformers_[-1].a_ = np.array(log_transform_params["a"])
            pipe.transformers_[-1].sign_ = np.array(log_transform_params["sign"])

        if normalize:
            scaler = StandardScaler()
            scaler.mean_ = np.array(standard_scaler_params["mean"])
            scaler.scale_ = np.array(standard_scaler_params["scale"])
            scaler.var_ = np.array(standard_scaler_params["scale"])**2
            scaler.n_features_in_ = len(standard_scaler_params["mean"])
            scaler.n_samples_seen_ = standard_scaler_params["n_samples_seen"]

        if pipe.power_transform != "none":
            pipe.transformers_.append(PowerTransformer(method=power_transform, standardize=normalize))
            #pipe.transformers_.append(QuantileTransformer(output_distribution='normal'))
            pipe.transformers_[-1].lambdas_ = np.array(power_transform_params["lambdas"])
            pipe.transformers_[-1].n_features_in_ = len(power_transform_params["lambdas"])
            if normalize:
                pipe.transformers_[-1]._scaler = scaler
        else:
            if normalize:
                pipe.transformers_.append(scaler)
            #if normalize:
            #    pipe.transformers_.append(scaler)




        pipe.is_fitted_ = True

        return pipe
        
    def _get_log_transformation(self, X):
        """ Get a y = log(x-a) transformer. 
        """

        # find features suitable for log transofrmation. 
        pattern = r"p\d_(.{1,5})"
        self.log_transform_features_ = []
        for i, feat in enumerate(self.features):
            match = re.findall(pattern, feat)
            if len(match)>0:
                if match[0] in ['real', 'mag']:
                    self.log_transform_features_.append(i)
        self.log_transform_features_ = np.array(self.log_transform_features_)

        if len(self.log_transform_features_) == 0:
            raise RuntimeError(f"The log transformer didn't find any 'real' or 'mag' features in {self.features}!")

        lt = LogTransformer(alpha=0.9).fit(X[:,self.log_transform_features_])
        X_transformed_lt = lt.transform(X[:,self.log_transform_features_])
        X_transformed = X.copy()
        X_transformed[:,self.log_transform_features_] = X_transformed_lt

        return X_transformed, lt

    def _get_power_transformation(self, X):
        """ Get a power transformer fitted to the data X and plot the transformation results. 
        """
        pt = PowerTransformer(standardize=self.normalize).fit(X)
        X_transformed = pt.transform(X)

        return X_transformed, pt
    

    def _get_standard_scaler(self, X):
        """ Get a standard scaler fitted to the data X and plot the transformation results. 
        """
        scaler = StandardScaler().fit(X)
        X_transformed = scaler.transform(X)
        
        return X_transformed, scaler
    

    def _plot_transformation(self, X_list, method_list):
        """ Plot a histogram comparison between transformed and non-transformed data
        """

        fig, axes = plt.subplots(len(X_list), X_list[0].shape[1], sharey=True, layout='constrained')
        axes = axes.reshape(len(X_list), X_list[0].shape[1])

        hist_kwargs = {"bins": 50}

        for i, X in enumerate(X_list):
            for j in range(X.shape[1]):
                axes[i,j].hist(X[:,j], color = cmap(i), **hist_kwargs)
            #axes[i,j].hist(X[:,i], color = cmap(i), **hist_kwargs)
                
            axes[i,0].set_ylabel("counts")
            axes[i, int(np.floor(X.shape[1]/2))].set_title(method_list[i])

        for j in range(X.shape[1]):
            axes[-1,j].set_xlabel(f'{self.features[j]}')
            
        fig.suptitle(f"Preprocessing pipeline on {self.feature_set} input data")

        if self.save:
            method_str = ""
            for m in method_list[1:]:
                method_str += f"{m}-"
            method_str = method_str[:-1]
            
            filepath = f"feature-trafo_{self.feature_set}_{method_str}.png"
            if self.tag is not None:
                filepath = self.tag + "_" + filepath
            filepath = os.path.join(self.dir_out, filepath)

            fig.set_size_inches(16.5, 9.5)
            fig.savefig(filepath)


    def fit_transform(self, X):
        """ Fit the pipeline to the data X and apply transformation to X. """

        self.n_features = X.shape[1]

        #X[:,1] = np.log(-X[:,1])
        #X[:,3] = np.log(-X[:,3])
        X_list = [X]
        method_list = ["orignal"]

        if self.log_transform:
            X_transformed, lt = self._get_log_transformation(X_list[-1])
            X_list.append(X_transformed)
            method_list.append(f'log-shift-transform')
            self.transformers_.append(lt)

        if self.power_transform != 'none':
            X_transformed, pt = self._get_power_transformation(X_list[-1])
            X_list.append(X_transformed)
            method_list.append(f'{self.power_transform}-power-transform')
            self.transformers_.append(pt)

        elif self.normalize:
            X_transformed, scaler = self._get_standard_scaler(X_list[-1])
            X_list.append(X_transformed)
            method_list.append(f'standard-scaling')
            self.transformers_.append(scaler)

        self._plot_transformation(X_list, method_list)

        return X_list[-1]


    def transform(self, X, sparse_column_indices=None):
        """ Apply the transformation of the preprocessing pipeline to the data X
        
        Parameters
        ----------
        sparce_column_indices : array-like
            Feature indices to apply the transformation to in case of sparse data.
        """

        # expand column-sparse sparse data
        if sparse_column_indices is not None:
            sparse_column_indices = np.array(sparse_column_indices).flatten()
            X_transform = np.zeros((X.shape[0], self.n_features))
            for i,j in enumerate(sparse_column_indices):
                X_transform[:,j] = X[:,i].flatten()
        else:
            X_transform = X.copy()


        #transform
        for trafo in self.transformers_:
            if isinstance(trafo, LogTransformer):
                #if logtrafo, only apply to suitable features
                for i in range(self.n_features):
                    if np.any(i==self.log_transform_features_) and not np.any(sparse_column_indices==i) and sparse_column_indices is not None:
                        idx_in_trafo = np.argwhere(i==self.log_transform_features_).flatten()
                        X_transform[:,i] = trafo.sign_[0,idx_in_trafo] * trafo.a_[0,idx_in_trafo] * 2
                X_transform[:,self.log_transform_features_] = trafo.transform(X_transform[:,self.log_transform_features_])
            else:
                X_transform = trafo.transform(X_transform)

        # reduce to column-sparse data
        if sparse_column_indices is not None:
            X_transform = X_transform[:,sparse_column_indices]

        return X_transform


    def inverse_transform(self, X, sparse_column_indices=None):
        """ Apply the inverse transformation of the preprocessing pipeline to the data X
        
        Parameters
        ----------
        sparce_column_indices : array-like
            Feature indices to apply the transformation to in case of sparse data.
        """
        
        # expand column-sparse sparse data
        if sparse_column_indices is not None:
            X_transform = np.zeros((X.shape[0], self.n_features))
            for i,j in enumerate(sparse_column_indices):
                X_transform[:,j] = X[:,i].flatten()
        else:
            X_transform = X.copy()

        #transform
        for trafo in reversed(self.transformers_):
            if isinstance(trafo, LogTransformer):
                #if logtrafo, only apply to suitable features
                X_transform[:, self.log_transform_features_] = trafo.inverse_transform(X_transform[:,self.log_transform_features_])
            else:
                X_transform = trafo.inverse_transform(X_transform)

        # reduce to column-sparse data
        if sparse_column_indices:
            X_transform = X_transform[:,sparse_column_indices]

        return X_transform
    


class PoleModel():
    """ A clase to fit predictive pole models. 
    """

    PREDEFINED_FEATURE_SETS = {
                     'ImRe5': [["p0_real", "p1_real", "p1_imag", "p2_real", "p2_imag"]],
                     'ImRe5GivenV': [["v_mean", "p0_real", "p1_real", "p1_imag", "p2_real", "p2_imag"], "v_mean"],
                     'AngMag5': [["p0_real", "p1_mag", "p1_ang", "p2_mag", "p2_ang"]],
                     'AngMag5GivenV': [["v_mean", "p0_real", "p1_mag", "p1_ang", "p2_mag", "p2_ang"], "v_mean"],
                     'Re1': [["p0_real"]],
                     'Re1GivenV': [["v_mean", "p0_real",], "v_mean"],
                     }
    
    REQUIRED_PATHS = ["filepath_partition", "filepath_sorted_poles", "dir_out"]

    SUBDIRS = {"output-dirname": "pole-modeling"}

    OUTPUT_FNAMES = {"gridsearch-results": "gridsearch",
                     "marginal-distributions": "distribution-model",
                     "model-export": "pole-model-params"}


    def __init__(self,
                 paths, 
                 gridsearch_selection_metric='NLL', 
                 normalization=False, 
                 power_transformation='yeo-johnson',
                 feature_set='ImRe5',
                 n_gmm_inits=100,
                 riderbike_model=None,
                 pole_table=None,
                 save=True,
                 from_data=True,
                 random_state=None):
        
        #data
        self.paths = self._check_paths(paths, save, from_data, pole_table is not None)
        self.riderbike_model = riderbike_model
        if from_data:
            self._load_data(pole_table)

        #features
        self.feature_set, self.features, self.feature_cond = self._check_feature_set(feature_set)

        #preprocessing
        self.normalize = normalization
        self.pt_type = power_transformation 
        self.pp_pipeline = PreprocessingPipeline(self.feature_set, 
                                    self.features, 
                                    normalize=self.normalize, 
                                    power_transform=self.pt_type, 
                                    save=save, 
                                    dir_out=self.paths['dir_out'],
                                    tag=self.riderbike_model)

        #model
        self._n_gmm_inits = n_gmm_inits
        self.is_fitted_ = False
        self.random_state = random_state

        #gridsearch
        self.gs_sel_metric = gridsearch_selection_metric
        
        #output
        self.save=save

    
    def _check_paths(self, paths, save, from_data, has_pole_table):
        """ Check that all required paths are supplied and exist. """

        required_paths = []
        if not has_pole_table and from_data:
            required_paths.append(self.REQUIRED_PATHS[1])
        if from_data:
            required_paths.append(self.REQUIRED_PATHS[0])
        if save:
            required_paths.append(self.REQUIRED_PATHS[2])

        for p in required_paths:
            if p not in paths:
                raise ValueError(f"Path to {p} missing in 'paths'. 'paths' must have at least {required_paths}.")
            if 'filepath' in p:
                if not os.path.isfile(paths[p]):
                    raise IOError(f"Can't find file '{p}' at {paths[p]}.")
            elif 'dir' in p:
                if paths[p] is None:
                    raise ValueError(f"Path {p} mustn't be None! Provide a valid path.")
                if not os.path.isdir(paths[p]):
                    raise IOError(f"Can't find directory '{p}' at {paths[p]}.")
        
        # if necessary, make output directory
        if not os.path.basename(os.path.normpath(paths['dir_out'])) == self.SUBDIRS["output-dirname"]:
            paths['dir_out'] = os.path.join(paths['dir_out'], self.SUBDIRS["output-dirname"])
        if not os.path.isdir(paths['dir_out']):
            os.makedirs(paths['dir_out'])

        return paths

    def _check_feature_set(self, feature_set):
        """Check that the selected feature set is valid."""
        
        valid_keys = list(self.PREDEFINED_FEATURE_SETS.keys())

        if isinstance(feature_set, str):
            if feature_set not in self.PREDEFINED_FEATURE_SETS:
                raise ValueError(f"If a String, 'feature_set' must be any of the predefined features {valid_keys}, not '{feature_set}'")
            
            feature_set_name = feature_set
            features = self.PREDEFINED_FEATURE_SETS[feature_set][0]
            if len(self.PREDEFINED_FEATURE_SETS[feature_set])>1:
                feature_cond = self.PREDEFINED_FEATURE_SETS[feature_set][1]
            else:
                feature_cond = ""
        else:
            raise NotImplementedError((f'Feature sets other then the predifined sets are '
                                       f'currently not implemented! Choose any of {valid_keys}'))
        
        self._make_tex_polelabels(features)
        
        return feature_set_name, features, feature_cond


    def _make_tex_polelabels(self, features):
        """Make pole labels in TeX maths format
        """
        self.tex_feature_labels = []
        for f in features:
            n = f[1]

            if n == '0':
                nstr = "2"
            if n == '1':
                nstr = "1, 3"
            if n == '2':
                nstr = "0, 4"

            if 'real' in f:
                l = r"$\Re(s_{"+nstr+r"})$"
            elif 'imag' in f:
                l = r"$|\Im(s_{"+nstr+r"})|$"
            elif 'ang' in f:
                l = r"$\varphi_"+str(n)+r"$"
            elif 'mag' in f:
                l = r"$r_"+str(n)+r"$"
            elif 'v_mean' in f:
                l = r"$\bar{v}$ [$m~s^{-1}$]"
            else:
                l = f
            self.tex_feature_labels.append(l)


    def _init_gmm(self, n_components, covariance_type):
        """Init a (conditional) GMM for the given hyperparamters.
        """ 

        gmm_kwargs = dict(n_init=self._n_gmm_inits,
                          n_components=n_components,
                          covariance_type=covariance_type)
        
        if self.feature_cond != '':
            gmm_kwargs["feature_index_given"] = self.features.index(self.feature_cond)
            return ConditionalGaussianMixture(**gmm_kwargs), score_conditional_gmm
        else:
            return GaussianMixture(**gmm_kwargs), score_gmm


    def _init_gmm_from_params(self, means, covariances, weights, random_state):
        """Init a (conditional) GMM from known parameters
        """

        if self.feature_cond != '':
            gmm = ConditionalGaussianMixture.from_parameters(means, covariances, weights, 
                                                             self.features.index(self.feature_cond),
                                                             random_state=random_state)
            score_func = score_conditional_gmm
            gmm.feature_indices_marginals = [i for i in np.arange(len(self.features)) if i != self.features.index(self.feature_cond)]
        else:
            gmm = GaussianMixture.from_parameters(means, covariances, weights, random_state=random_state)
            score_func = score_gmm

        return gmm, score_func


    def _check_conditional_features(self, conditional_features):
        """ Check if the conditional features appear in the given features.

        UNUSED.
        """
        
        missing_feature_error = ValueError(("The conditional features must appear in the feature list" 
                                           "of the corresponding feature set or, for non-conditional" 
                                           "models, be empty strings!"))
        
        for key in conditional_features:
            if key not in self.features:
                continue
            if conditional_features[key] not in self.features[key]:
                raise missing_feature_error
        
        for key in self.features:
            if key not in conditional_features:
                conditional_features[key] = ''

        return conditional_features
    
    
    def _load_data(self, pole_table):
        """ Load the data.
        """

        # load fixed partitioning
        self.partition = read_yaml(self.paths["filepath_partition"])

        if pole_table is None:
            # load poles
            pole_table = pd.read_csv(self.paths["filepath_sorted_poles"])

        if 'outliers_all' in pole_table:
            outliers = pole_table['outliers_all'] 
        else:
            outliers = pole_table['outliers'] 
        
        self.pole_table_outliers = pole_table[outliers]
        self.pole_table_inliers = pole_table[np.logical_not(outliers)]


    def _calibrate_variance_scale(self, X_train):

        var_scale = np.linspace(0.2, 1.0, 25)
        n_calib_samples = 10000
        
        alpha = 0.05
        n_quantile = int(round(X_train.shape[0] * (alpha)))
        if n_quantile == 0:
            raise RuntimeError(f"Not enough samples for alpha={alpha} calibration!")
        
        calib_score = np.zeros_like(var_scale)

        gmm_0, score_func = self._init_gmm(n_components=self.hyperparameters_['n_components'], 
                                    covariance_type=self.hyperparameters_['cov_type'], 
                                    var_scale=1.0)
        gmm_0.fit(X_train)
        nll_train = gmm_0.score_samples(X_train)
        worst_samples_train = np.argsort(nll_train)[-n_quantile:]
        nll_limit = np.min(nll_train[worst_samples_train])

        for i, s in enumerate(var_scale): 
            gmm, score_func = self._init_gmm(n_components=self.hyperparameters_['n_components'], 
                                    covariance_type=self.hyperparameters_['cov_type'], 
                                    var_scale=s)
            gmm = gmm.fit(X_train)   
            X_calib, _ = gmm.sample(n_samples=n_calib_samples)

            nll_calib = gmm.score_samples(X_calib)
            calib_score[i] = np.sum(nll_calib>nll_limit)/n_calib_samples

        best_calib = np.argmin(np.abs(calib_score-alpha))
        s_best = var_scale[best_calib]
        calib_score_best = calib_score[best_calib]

        print(f"    Variance Scale calibration at alpha={alpha} ({n_quantile} worst training samples): s={s_best}, score={calib_score_best}")

        self.hyperparameters_['var_scale'] = s_best
        self.scores_val_['variance_scale_calibration'] = calib_score_best

    def get_datasets(self):

        # get dataset
        X = self.pole_table_inliers[self.features].to_numpy()

        idx_train = self.pole_table_inliers['sample_id'].isin(self.partition["train"])
        idx_test = self.pole_table_inliers['sample_id'].isin(self.partition["test"])
        
        X = self.pp_pipeline.fit_transform(X)

        X_train = X[idx_train,:]
        X_test = X[idx_test,:]

        self.n_features_ = X_train.shape[1]
        self.n_samples_test_ = X_test.shape[0]
        self.n_samples_train_ = X_train.shape[0]

        return X_train, X_test
        

    def fit_optimize(self, 
                     range_gmm_components=[1,5], 
                     k_crossval=10,
                     covariance_types=["full", "tied", "diag", "spherical"]):
        """ Fit the pole model. Finds optimal hyperparmeters within the given ranges using cross-validation."""

        print(f"Fitting {self.feature_set} pole model on {self.riderbike_model} poledata:")
        self.k_crossval_ = 10
        
        #score dict
        model_scores = {
            "normalization": [],
            "power-transform": [],
            "cov_type": [],
            "n_components": [],
            "BIC": [],
            "AIC": [],
            "NLL": []}

        # get data
        X_train, X_test = self.get_datasets()


        # Grid-Search based model optimization
        for cov_type in covariance_types:
            print(" "*100, end="\r")
            for n in range(range_gmm_components[0], range_gmm_components[1]):
                print(f"    Running Gridsearch: covariance_type = {cov_type}, n_components = {n}", end="\r")
                # fit and validate a Gaussian Mixture Model with two components using cross-validation
                gmm, score_func = self._init_gmm(n, cov_type)
                scores = cross_validate(gmm, X_train, scoring=score_func, cv=k_crossval, error_score='raise')
                
                model_scores["normalization"].append(self.normalize)
                model_scores["power-transform"].append(self.pt_type)
                model_scores["n_components"].append(n)
                model_scores["BIC"].append(np.mean(scores["test_BIC"]))
                model_scores["AIC"].append(np.mean(scores["test_AIC"]))
                model_scores["NLL"].append(np.mean(scores["test_NLL"]))
                model_scores["cov_type"].append(cov_type)

        # Identify best hyperparamters
        self.gridsearch_scores_ = pd.DataFrame.from_dict(model_scores)
        best = self.gridsearch_scores_[model_scores[self.gs_sel_metric]==np.min(model_scores[self.gs_sel_metric])]
        self.scores_val_ = best[["BIC", "AIC", "NLL"]].iloc[0].to_dict()
        self.hyperparameters_ = best[["n_components", "cov_type"]].iloc[0].to_dict()

        # fit and test a model on the full training dataset with the best hyperparameters

        self.gmm_, score_func = self._init_gmm(n_components=self.hyperparameters_['n_components'], 
                                     covariance_type=self.hyperparameters_['cov_type'])
        self.gmm_ = self.gmm_.fit(X_train)   
        self.is_fitted_ = True     

        self.scores_test_ = score_func(self.gmm_, X_test)

        # plot best model
        self.plot_marginals(X_train, X_test, k_crossval)
        self.plot_gridsearch()

        print(f"    Finished gridsearch with best results covariance_type={self.hyperparameters_["cov_type"]} and n_components={self.hyperparameters_["n_components"]} achieving NLL={self.scores_val_["NLL"]:.4f}")

        return self
    

    def sample(self, n_samples=1, X_given=None, shuffle=True):
        """ Sample from the fitted distribution. 

        If PoleModel is a conditional model. A value to be conditioned on must be given. 

        Parameters
        ----------
        n_samples : int, optional
            Number of samples to be drawn
        X_given : list-like
            List of n_given values of the conditional feature to be conditioned on. 
        shuffle : bool, optional
            Shuffle the returned samples. Default is True

        Returns
        -------
        samples : array-like
            Array of samples shaped [n_samples, n_features]. If conditional and n_given>1, the shape is
            Array of samples shaped [n_given, n_samples, n_features].
        """

        if self.feature_cond != '':
            if X_given is None:
                raise ValueError("Specify values for {self.feature_cond} to be conditioned on to sample poles!")
            X_given = np.array(X_given)
            x_given_temp = np.zeros((X_given.size, self.n_features_))
            x_given_temp[:,self.features.index(self.feature_cond)] = X_given.flatten()
            x_given_temp = self.pp_pipeline.transform(x_given_temp, sparse_column_indices=[self.features.index(self.feature_cond)])
            X_given = x_given_temp[:,self.features.index(self.feature_cond)]

            samples, labels = self.gmm_.sample(n_samples=n_samples, X_given=X_given)
        else:
            samples, labels = self.gmm_.sample(n_samples)

        if not np.all(np.isfinite(samples)):
            raise RuntimeError("Sampling error!")
        
        indices = [i for i, f in enumerate(self.features) if f != self.feature_cond]
        samples_out = self.pp_pipeline.inverse_transform(samples, sparse_column_indices=indices)
        
        # the sampled values may violate the valid range of the yeo-johnson inverse transform. Resample invalid values. 
        i = 0
        while not np.all(np.isfinite(samples_out)):
            missing_samples = np.logical_not(np.all(np.isfinite(samples_out), axis=1))
            n_missing_samples = np.sum(missing_samples)

            if self.feature_cond != '':
                new_samples, new_labels = self.gmm_.sample(n_samples=n_missing_samples, X_given=X_given)
            else:
                new_samples, new_labels = self.gmm_.sample(n_missing_samples)

            new_samples = self.pp_pipeline.inverse_transform(new_samples, sparse_column_indices=indices)
            samples_out[missing_samples, :] = new_samples
            labels[missing_samples] = new_labels

            i+=1
            if i>100:
                raise RuntimeError("Sampling error!")
    
        if not np.all(np.isfinite(samples_out)):
            raise RuntimeError("Sampling error!")

        #shuffle
        if shuffle:
            rng = np.random.default_rng(seed=self.random_state)
            shuffle_idx = np.arange(n_samples)
            rng.shuffle(shuffle_idx)
            
            samples_out = samples_out[shuffle_idx,:]
            labels = labels[shuffle_idx]

        if not np.all(np.isfinite(samples_out)):
            raise RuntimeError("Sampling error!")

        return samples_out, labels
    

    def sample_poles(self, n_samples=1, X_given=None, ensure_stable=True):
        """ Sample poles from the fitted distribution. 

        If PoleModel is a conditional model. A value to be conditioned on must be given. 

        Parameters
        ----------
        n_samples : int, optional
            Number of samples to be drawn
        X_given : list-like
            List of n_given values of the conditional feature to be conditioned on. 

        Returns
        -------
        samples : array-like
            Array of pole samples shaped [n_samples, n_poles]. 
        """

        if n_samples == 0:
            return np.array([[]]), np.array([])


        if X_given is not None:
            if not isinstance(X_given, float):
                raise TypeError(f"X_given must be 'float', not '{type(X_given).__name__}'!")
            
        features = [f for f in self.features if f != self.feature_cond]
        feature_indices = [i for i, f in enumerate(self.features) if f != self.feature_cond]

        if 'AngMag' in self.feature_set:
            feat = 'AngMag'
        else:
            feat = 'ImRe'
            
        def _sample(n):
            samples, labels = self.sample(n_samples=n, X_given=[X_given])
            pole_table = pd.DataFrame(samples, columns=features)
            pole_array = polefeaturetable_to_polearray(pole_table, feat)
            return pole_array, labels

        # draw initial sample
        pole_array, labels = _sample(n_samples)

        if ensure_stable:
            n_iter = 0
            n_iter_max = 1000
            while np.any(np.real(pole_array)>0):
                mask_unstable = np.any(np.real(pole_array)>0, axis=1)
                extra_poles = _sample(np.sum(mask_unstable))
                pole_array[mask_unstable,:] = extra_poles
                n_iter+=1

                if n_iter > n_iter_max:
                    raise TimeoutError(f"Couldn't find {n_samples} stable poles after {n_iter_max} draws!")

        return pole_array, labels

    def get_component_means(self, X_given=None):
        """ Return the component means. 

        If PoleModel is a conditional model. A value to be conditioned on can be given.

        Parameters
        ----------
        X_given : list-like
            List of n_given values of the conditional feature to be conditioned on. 

        Returns
        -------
        mean_components : array-like
            Array of mean_features shaped [n_samples, n_features(, n_given)]. If X_given is None, the array is 2D.  
        x_cond : array like
            Array of conditional values corresponding to the component means. Only returned if the pole model is conditional
            but no values to be conditioned on are given. 
        """

        if X_given is not None:
            if not isinstance(X_given, (float, list, tuple, np.ndarray)):
                raise TypeError(f"X_given must be 'float' or 'array-like', not '{type(X_given).__name__}'!")
            if isinstance(X_given, float):
                X_given = np.array([X_given])
            else:
                X_given = np.array([X_given]).flatten()
            
            x_given_temp = np.zeros((X_given.size, self.n_features_))
            x_given_temp[:,self.features.index(self.feature_cond)] = X_given.flatten()
            x_given_temp = self.pp_pipeline.transform(x_given_temp, sparse_column_indices=[self.features.index(self.feature_cond)])
            X_given = x_given_temp[:,self.features.index(self.feature_cond)]
            
        features = [f for f in self.features if f != self.feature_cond]
        feature_indices = [i for i, f in enumerate(self.features) if f != self.feature_cond]

        if 'AngMag' in self.feature_set:
            feat = 'AngMag'
        else:
            feat = 'ImRe'

        if X_given is not None:
            component_means = []
            for x_given in X_given:
                means_x = self.gmm_._get_conditional_gmm(x_given).means_.reshape((-1,len(features)))
                means_x = self.pp_pipeline.inverse_transform(means_x, sparse_column_indices=feature_indices)

                #pole_table = pd.DataFrame(means_x, columns=features)
                #pole_array = polefeaturetable_to_polearray(pole_table, feat)

                component_means.append(means_x)

            component_means = np.array(component_means).transpose((1, 2, 0))
        else:
            component_means = self.gmm_.means_
            component_means = self.pp_pipeline.inverse_transform(component_means)
            #pole_table = pd.DataFrame(means, columns=features)
            #component_means = polefeaturetable_to_polearray(pole_table, feat)

            if self.feature_cond != '':
                x_cond = component_means[:,self.features.index(self.feature_cond)]
                component_means = component_means[:,feature_indices]
                return component_means, x_cond

        return component_means

    def get_component_mean_poles(self, X_given=None):
        """ Return the component means as complex poles. 

        If PoleModel is a conditional model. A value to be conditioned on can be given.

        Parameters
        ----------
        X_given : list-like
            List of n_given values of the conditional feature to be conditioned on. 

        Returns
        -------
        mean_poles : array-like
            Array of mean poles shaped [n_samples, n_poles(, n_given)]. If X_given is None, the array is 2D.  
        x_cond : array like
            Array of conditional values corresponding to the component means. Only returned if the pole model is conditional
            but no values to be conditioned on are given. 
        """

        if self.feature_cond != '' and X_given is None:
            component_means, x_cond = self.get_component_means(X_given=X_given)
        else:
            component_means = self.get_component_means(X_given=X_given)
        mean_poles = np.zeros_like(component_means, dtype=complex)

        features = [f for f in self.features if f != self.feature_cond]
        feature_indices = [i for i, f in enumerate(self.features) if f != self.feature_cond]

        if 'AngMag' in self.feature_set:
            feat = 'AngMag'
        else:
            feat = 'ImRe'

        if component_means.ndim > 2:
            for i in range(component_means.shape[2]):
                pole_table = pd.DataFrame(component_means[:,:,i], columns=features)
                mean_poles[:,:,i] = polefeaturetable_to_polearray(pole_table, feat)
        else:
            pole_table = pd.DataFrame(component_means, columns=features)
            mean_poles = polefeaturetable_to_polearray(pole_table, feat)

        if self.feature_cond != '' and X_given is None:
            return mean_poles, x_cond
        else:
            return mean_poles
        

    def get_component_mean_function_params(self):
        """ Return the component means as a function of speed

        WARNING: This will be wrong if the means are not linear over speed! Check the marginal distribution plot 
        to verify!


        Returns
        ------
        array_like
            An array of the parameters of function linear in speed is returned. The array is shaped
            (n_components, n_features, 2), with the last dimension representing intercept and coefficient. 
        """

        if isinstance(self.gmm_, ConditionalGaussianMixture):
            speeds_cond = np.linspace(1.5,5.5,250)
            means = self.get_component_means(speeds_cond)

            regs = []
            scores = []
            print("Fitting a linear function of speed to the component means.")
            print(f"score per component: ")
            if means.ndim != 3:
                raise NotImplementedError("Not implemented for models with n_components=1!")
            for i in range(means.shape[0]):
                means_i = means[i, :, :].T
                reg = LinearRegression().fit(speeds_cond.reshape(-1,1), means_i)
                score = reg.score(speeds_cond.reshape(-1,1), means_i)
                scores.append(score)
                regs.append(reg)
                print(f" component {i}: R2 = {score:.2f}")

            if np.any(np.array(scores) < 0.9):
                print(f"   Fit resulted in an unsatifactory R2. Confirm that the speed"
                      f" dependency of the component means in linear by looking at the plot of the 2D marignals!")
                
            return np.stack([np.c_[reg.intercept_, reg.coef_.flatten()] for reg in regs], axis=0)
        else:
            means = self.get_component_means()
            return means


    def plot_gridsearch(self):
        """ Plot the gridsearch model selection results.
        """

        if not self.is_fitted_:
            raise RuntimeError("Model must be fitted before plotting by calling 'fit_optimize()")

        # get params
        covariance_types = np.unique(self.gridsearch_scores_['cov_type'])
        metrics = np.unique(list(self.scores_val_.keys()))

        # make axes
        fig, axes = plt.subplots(1, metrics.size, layout="constrained")

        # plot results
        for metric, ax in zip(metrics, axes):
            for i, ctype in enumerate(covariance_types):
                col = cmap.colors[i]      
                sel = self.gridsearch_scores_['cov_type'] == ctype
                results = np.array(self.gridsearch_scores_[sel][["n_components", metric]])
                ax.plot(results[:,0], results[:,1], color=col, label = ctype)

            # mark best
            ax.plot([self.hyperparameters_['n_components']], [self.scores_val_[metric]], color=tudcolors.get("rood"), marker="o")
            ax.annotate(f'{self.scores_val_[metric]:.2f}',
                xy=(self.hyperparameters_['n_components'], self.scores_val_[metric]), 
                horizontalalignment='left',
                verticalalignment='bottom')
            
            ax.set_title(metric)
            ax.set_ylabel("score")
            ax.set_xlabel("n_components")

        axes[0].legend()
        fig.suptitle(f'Grid Search GMM {self.feature_set} Model Selection: {self.riderbike_model}\n normalized: {self.normalize}, power-transfrom: {self.pt_type}')

        if self.save:
        
            if self.riderbike_model is not None:
                filepath = self.riderbike_model + "_"
            else:
                filepath = ""

            filepath = os.path.join(self.paths["dir_out"], filepath+f"{self.feature_set}_{self.OUTPUT_FNAMES["gridsearch-results"]}.png")
            fig.set_size_inches(16.5, 9.5)
            fig.savefig(filepath)

    def plot_marginals(self, X_train, X_test, k_crossval, marginals_2d = True, marginals_1d = True, plot_for_paper=False, 
                       sct_train_style=dict(s=5, color='black'), 
                       sct_test_style=dict(s=5, color=tudcolors.get('roze')),
                       sct_means_style=dict(s=5, color=tudcolors.get('rood')),
                       cond_density_style=dict(linewidth=1, color=tudcolors.get("donkerblauw")),
                       grid_style=None):
        """ Plot the marignal (and conditional) distributions of the fitted model."""

        if not self.is_fitted_:
            raise RuntimeError("Model must be fitted before plotting by calling 'fit_optimize()")
        
        n_features = self.gmm_.means_.shape[1]

        if marginals_2d and n_features >= 2:
            fig, ax = self._plot_2d_marginals(X_train, X_test, k_crossval, plot_for_paper, sct_train_style=sct_train_style,
                                              sct_test_style=sct_test_style, sct_means_style=sct_means_style, 
                                              cond_density_style=cond_density_style, grid_style=grid_style)
            #self._plot_2d_marginals(X_train, self.pp_pipeline.transform(self.sample(10000)), k_crossval=10)
        if marginals_1d:
            fig, ax = self._plot_1d_marginals(X_train, X_test, k_crossval)

        return fig, ax
        
    def _plot_1d_marginals(self, X_train, X_test, k_crossval):
        """ Plot the 1d marignal (and conditional) distributions of the fitted model."""

        # create gridspec 
        n_features = self.gmm_.means_.shape[1]
        n_plotsperrow_max = 8
        n_columns = min(n_features, n_plotsperrow_max)
        n_rows = int(np.ceil(n_features/n_plotsperrow_max))

        fig = plt.figure(layout="constrained")
        gs = GridSpec(n_rows, n_columns, figure=fig)
        axes = []

        # rescale data
        X_train_rescaled = self.pp_pipeline.inverse_transform(X_train)
        if not X_test is None:
            X_test_rescaled = self.pp_pipeline.inverse_transform(X_test)

        for i in range(n_features):
            ax = fig.add_subplot(gs[i//n_columns, i%n_columns])
            axes.append(ax)

            # grid for surface plot
            xlim = (np.min(X_train[:, i]) - 1, np.max(X_train[:, i]) + 1)

            # accumulate density
            locations, density = self.gmm_.eval_1d_marginal_pdf(xlim, i)   

            # rescale
            locations = self.pp_pipeline.inverse_transform(locations[:,np.newaxis], sparse_column_indices=[i,]).flatten()

            # plot
            ax.hist(X_train_rescaled[:, i], color='black', density=True, bins=100, label='training samples')
            ax.plot(locations, density, color=tudcolors.get('blauw'), label='model distribution')
            if not X_test is None:
                ax.scatter(X_test_rescaled[:, i], self.gmm_.eval_1d_marginal_pdf_samples(X_test[:, i], i)[1], s=5, color=tudcolors.get('roze'), zorder=100, label='test samples')

            #set plot limits
            location_limits = [np.nanmin(locations), np.nanmax(locations)] 
            
            if not X_test is None:
                minplot = min((np.nanmin(X_train_rescaled[:, i]) - 1, np.nanmin(X_test_rescaled[:, i]) - 1))
                maxplot = max((np.nanmax(X_train_rescaled[:, i]) + 1, np.nanmax(X_test_rescaled[:, i]) + 1))
            else:
                minplot = np.nanmin(X_train_rescaled[:, i]) - 1
                maxplot = np.nanmax(X_train_rescaled[:, i]) + 1

            minplot = max(minplot, location_limits[0])
            maxplot = min(maxplot, location_limits[1])

            ax.set_xlim(minplot, maxplot)
            ax.set_xlabel(self.features[i])

        axes[0].set_ylabel("density")
        axes[-1].legend()

        # title
        valstr = f"{k_crossval:<2}-CROSSVAL"
        teststr = f"TEST"
        nparams = self.gmm_._n_parameters()
        fig.suptitle((f"Best {self.feature_set} Gaussian-Mixture-Model: {self.riderbike_model}\n"
                f"n_components = {self.hyperparameters_['n_components']}, preprocessing: {self.pp_pipeline.method_str}, {self.hyperparameters_['cov_type']}, {nparams} params\n"
                f"      {valstr:<11}  {teststr:<5}\n"
                f"BIC:  {self.scores_val_['BIC']:<11.2f}  {self.scores_test_['BIC']:<5.2f}\n"
                f"AIC:  {self.scores_val_['AIC']:<11.2f}  {self.scores_test_['AIC']:<5.2f}\n"
                f"NLL:  {self.scores_val_['NLL']:<11.2f}  {self.scores_test_['NLL']:<5.2f}\n"))
        
        if self.save:
        
            if self.riderbike_model is not None:
                filepath = self.riderbike_model + "_"
            else:
                filepath = ""

            filepath = os.path.join(self.paths["dir_out"], filepath+f"{self.feature_set}_1d-{self.OUTPUT_FNAMES["marginal-distributions"]}.png")
            fig.set_size_inches(max(3.5*n_columns, 8), max(3.5 * n_rows, 6.5))
            fig.savefig(filepath)

        return fig, ax


    def _plot_2d_marginals(self, X_train=None, X_test=None, k_crossval=None, plot_for_paper=False,
                           sct_train_style=dict(s=5, color='black'), 
                           sct_test_style=dict(s=5, color=tudcolors.get('roze')),
                           sct_means_style=dict(s=5, color=tudcolors.get('rood')),
                           cond_density_style=dict(linewidth=1, color=tudcolors.get("donkerblauw")),
                           grid_style=None):
        """ Plot the 2d marignal (and conditional) distributions of the fitted model."""

        # get feature index pairs
        n_features = self.gmm_.means_.shape[1]
        feature_pairs = np.array(np.triu_indices(n_features, k=1))
        feature_pairs = np.array(feature_pairs).T.tolist()
        n_pairs = len(feature_pairs)

        # create gridspec 
        n_plotsperrow_max = 5
        n_columns = min(n_pairs, n_plotsperrow_max)
        n_rows = int(np.ceil(n_pairs/n_plotsperrow_max))

        fig = plt.figure(layout="constrained")
        gs = GridSpec(n_rows, n_columns, figure=fig)
        axes = []

        # rescale data
        if X_train is not None:
            X_train_rescaled = self.pp_pipeline.inverse_transform(X_train)
        if X_test is not None:
            X_test_rescaled = self.pp_pipeline.inverse_transform(X_test)

        means_rescaled = self.pp_pipeline.inverse_transform(self.gmm_.means_)
        speeds_cond = np.linspace(1.5,5.5,50)
        if isinstance(self.gmm_, ConditionalGaussianMixture):
            means_speed_cond = self.get_component_means(speeds_cond)

        for idx, (i, j) in enumerate(feature_pairs):
            ax = fig.add_subplot(gs[idx//n_columns, idx%n_columns])
            axes.append(ax)

            #grid
            if isinstance(grid_style, dict):
                ax.grid(**grid_style)

            # grid for surface plot
            xlim = [np.min(X_train[:, i]) - 1, np.max(X_train[:, i]) + 1]
            ylim = [np.min(X_train[:, j]) - 1, np.max(X_train[:, j]) + 1]

            #pattern=r"p\d_(.{1,5})"
            #for klim, k in zip([xlim, ylim], [i,j]):
            #    if re.findall(pattern, self.features[k])[0] in ["real"]:
            #        klim[1] = 0
            #    if re.findall(pattern, self.features[k])[0] in ["imag", "mag"]:
            #        klim[0] = 0
            #    if re.findall(pattern, self.features[k])[0] in ["ang"]:
            #        klim[0] = np.pi/2

            # accumulate density
            locations, density = self.gmm_.eval_2d_marginal_pdf(xlim, ylim, i, j)
            N = int(np.sqrt(density.size))

            # rescale
            locations = self.pp_pipeline.inverse_transform(locations, sparse_column_indices=[i,j])
            locations = locations.reshape(N, N, 2)
            density = density.reshape(N, N)

            # plot distribution
            ax.contourf(locations[:,:,0], locations[:,:,1], density, levels=30, 
                        cmap=tudcolors.colormap(name="turkoois"), alpha=0.8)
            ax.contour(locations[:,:,0], locations[:,:,1], density, levels=30, colors='gray', linewidths=0.2)
            
            # plot samples
            if X_train is not None:
                ax.scatter(X_train_rescaled[:, i], X_train_rescaled[:, j], **sct_train_style, zorder=1000)
            if X_test is not None:
                ax.scatter(X_test_rescaled[:, i], X_test_rescaled[:, j], **sct_test_style, zorder=2000)
            
            # plot means
            for k in range(means_rescaled.shape[0]):
                ax.scatter(means_rescaled[k, i], means_rescaled[k, j], **sct_means_style, zorder=3000)
                ax.annotate(str(k), xy=(means_rescaled[k, i], means_rescaled[k, j]+(np.mean(means_rescaled[:, j])*np.sign(means_rescaled[k, j])*0.1)), color=sct_means_style['color'], zorder=3000)

            #set plot limits
            location_limits = [(np.nanmin(locations[:, :, 0]), np.nanmax(locations[:, :, 0])), 
                               (np.nanmin(locations[:, :, 1]), np.nanmax(locations[:, :, 1]))]

            for k, func, loc_lim in zip([i,j],[ax.set_xlim, ax.set_ylim], location_limits):
                if not X_test is None:
                    minplot = min((np.nanmin(X_train_rescaled[:, k]) - 1, np.nanmin(X_test_rescaled[:, k]) - 1))
                    maxplot = max((np.nanmax(X_train_rescaled[:, k]) + 1, np.nanmax(X_test_rescaled[:, k]) + 1))
                else:
                    minplot = np.nanmin(X_train_rescaled[:, k]) - 1
                    maxplot = np.nanmax(X_train_rescaled[:, k]) + 1

                minplot = max(minplot, loc_lim[0])
                maxplot = min(maxplot, loc_lim[1])

                if plot_for_paper:
                    if 'real' in self.features[k]:
                        maxplot = 0
                    if 'imag' in self.features[k]:
                        minplot = 0
                    if 'ang' in self.features[k]:
                        minplot = np.pi/2
                    if 'mag' in self.features[k]:
                        minplot = 0

                func(minplot, maxplot)


            # plot conditional
            features = [f for f in self.features if f != self.feature_cond]
            if isinstance(self.gmm_, ConditionalGaussianMixture) and self.feature_cond in [self.features[i], self.features[j]]:
                if self.features[i] == self.feature_cond:
                    lim = ylim
                    idx_marg = j   
                else:
                    lim = xlim
                    idx_marg = i
                    
                speeds = (np.array([8,11,14])/3.6).reshape((3,1))
                speeds_scaled = self.pp_pipeline.transform(speeds, sparse_column_indices=[self.features.index('v_mean')])
                
                for v, v_scaled in zip(speeds, speeds_scaled):

                    locations, density = self.gmm_.eval_conditional_marginal_pdf(lim, v_scaled, idx_marg)
                    locations = self.pp_pipeline.inverse_transform(locations[:,np.newaxis], sparse_column_indices=[idx_marg])
                    loc_ext = [0, np.min(np.abs(locations)) * np.sign(locations).flatten()[0]]

                    if self.features[i] == self.feature_cond:
                        ax.plot(density+v, locations, **cond_density_style)
                        ax.plot(np.zeros_like(density)+v, locations, linestyle='--', **cond_density_style)
                        
                        # extend density line for illegal values
                        ax.plot([v,v], loc_ext, **cond_density_style)
                    else:
                        ax.plot(locations, density+v,  **cond_density_style)
                        ax.plot(locations, np.zeros_like(density)+v, linestyle='--', **cond_density_style)

                        # extend density line for illegal values
                        ax.plot(loc_ext, [v,v], **cond_density_style)

                #plot speed means
                for k in range(means_rescaled.shape[0]):
                    if self.features[i] == self.feature_cond:
                        ax.plot(speeds_cond, means_speed_cond[k, features.index(self.features[j]), :], color=sct_means_style['color'], linestyle='--', linewidth=cond_density_style['linewidth'])
                    else:
                        ax.plot(means_speed_cond[k, j, :], speeds_cond, color=sct_means_style['color'], linestyle='--', linewidth=cond_density_style['linewidth'])
            
            #axis labels
            if plot_for_paper:
                ax.set_xlabel(self.tex_feature_labels[i], labelpad=0)
                ax.set_ylabel(self.tex_feature_labels[j], labelpad=1)              
                ax.tick_params(axis='x', pad=1)
                ax.tick_params(axis='y', pad=1)
            else:            
                ax.set_xlabel(self.features[i])
                ax.set_ylabel(self.features[j])


        # title
        if not plot_for_paper:
            valstr = f"{k_crossval:<2}-CROSSVAL"
            teststr = f"TEST"
            nparams = self.gmm_._n_parameters()
            fig.suptitle((f"Best {self.feature_set} Gaussian-Mixture-Model: {self.riderbike_model}\n"
                    f"n_components = {self.hyperparameters_['n_components']}, preprocessing: {self.pp_pipeline.method_str}, {self.hyperparameters_['cov_type']}, {nparams} params\n"
                    f"      {valstr:<11}  {teststr:<5}\n"
                    f"BIC:  {self.scores_val_['BIC']:<11.2f}  {self.scores_test_['BIC']:<5.2f}\n"
                    f"AIC:  {self.scores_val_['AIC']:<11.2f}  {self.scores_test_['AIC']:<5.2f}\n"
                    f"NLL:  {self.scores_val_['NLL']:<11.2f}  {self.scores_test_['NLL']:<5.2f}\n"))
        
        if self.save and not plot_for_paper:
            if self.riderbike_model is not None:
                filepath = self.riderbike_model + "_"
            else:
                filepath = ""

            filepath = os.path.join(self.paths["dir_out"], filepath+f"{self.feature_set}_2d-{self.OUTPUT_FNAMES["marginal-distributions"]}.png")
            fig.set_size_inches(16.5, 9.5)
            fig.savefig(filepath)

        return fig, ax


    def export_to_yaml(self):
        """ Export this model as yaml.
        """

        if not self.is_fitted_:
            raise RuntimeError(f"Fit the pole model to data to create parameters that can be saved!")

        # preprocessing pipeline
        preprocessing_pipe = dict(
            power_transform = self.pp_pipeline.power_transform,
            normalize=self.pp_pipeline.normalize, 
            log_transform=self.pp_pipeline.log_transform)
        
        power_transform_params = {}
        log_transform_params = {}

        for trafo in self.pp_pipeline.transformers_:
            if isinstance(trafo, PowerTransformer):
                power_transform_params["lambdas"] = trafo.lambdas_.tolist()
                if self.pp_pipeline.power_transform != 'none' and self.pp_pipeline.normalize:
                    scaler = trafo._scaler
            elif isinstance(trafo, LogTransformer):
                log_transform_params['a'] = trafo.a_.tolist()
                log_transform_params['sign'] = trafo.sign_.tolist()
                log_transform_params['log_transform_features'] = self.pp_pipeline.log_transform_features_.tolist()
            elif isinstance(trafo, StandardScaler):
                scaler = scaler

        if self.pp_pipeline.normalize:
            standard_scaler_params = dict(
                mean = scaler.mean_.tolist(),
                scale = scaler.scale_.tolist(),
                n_samples_seen = int(scaler.n_samples_seen_))
        else:
            standard_scaler_params = {}
        
        preprocessing_pipe['power_transform_params'] = power_transform_params
        preprocessing_pipe['standard_scaler_params'] = standard_scaler_params
        preprocessing_pipe['log_transform_params'] = log_transform_params
        

        # gaussian mixture model 
        gmm = dict(
            means = self.gmm_.means_.tolist(),
            covariances = self.gmm_.get_full_covariancematrix().tolist(), 
            weights = self.gmm_.weights_.tolist(),
            scores_val = self.scores_val_,
            scores_test = {k: float(v) for k, v in self.scores_test_.items()},
            n_samples_test = self.n_samples_test_,
            n_samples_train = self.n_samples_train_,
            n_features = self.gmm_.means_.shape[1],
            n_components = self.gmm_.means_.shape[0],
            k_crossval = self.k_crossval_,
            covariance_type = self.gmm_.covariance_type
        )

        presets = dict(
            feature_set = self.feature_set,
            features = self.features,
            gridsearch_selection_metric=self.gs_sel_metric, 
            n_gmm_inits=self._n_gmm_inits,
            riderbike_model=self.riderbike_model,
        )
        metadata = dict(
            data_created=str(datetime.now())
        )

        data = dict(
            presets = presets, 
            gmm_data = gmm,
            preprocessing_pipeline=preprocessing_pipe,
            metadata = metadata
        )

        #save to yaml
        if self.riderbike_model is not None:
            filepath = self.riderbike_model + "_"
        else:
            filepath = ""

        filepath = os.path.join(self.paths["dir_out"], filepath+f"{self.feature_set}_{self.OUTPUT_FNAMES["model-export"]}.yaml")

        with open(filepath, "w") as f:
            yaml.dump(data, f) 

    def import_from_yaml(filepath, save=False, dir_out="", random_state=None):
        """ Create a PoleModel object from a yaml image as created by PoleModel.export_to_yaml().

        Results in a fitted PoleModel object that can be used to sample new poles. 

        Parameters
        ----------
        filepath : str
            File path to the yaml image.
        save : bool, optional
            Configure to pole model to save any output it creates to file.
        dir_out : None, optional
            Configure the output director of this pole model. 
        random_state : None, optional
            If an integer, use the integer as fixed random seed for the generation of random numbers. 

        Returns:
        --------
        pm : PoleModel
            A fitted pole-model object
        """

        with open(filepath, "r") as f:
            data = yaml.safe_load(f)

        # PoleModel object
        paths = dict(dir_out=dir_out)
        pm = PoleModel(paths, 
                       gridsearch_selection_metric= data["presets"]["gridsearch_selection_metric"],
                       normalization=data["preprocessing_pipeline"]["normalize"],
                       power_transformation=data["preprocessing_pipeline"]["power_transform"],
                       feature_set=data['presets']['feature_set'],
                       n_gmm_inits=data['presets']['n_gmm_inits'], 
                       riderbike_model=data['presets']['riderbike_model'], 
                       save=save,
                       from_data=False,
                       random_state=random_state)
        
        # Configure Preprocessing Pipeline
        kwargs_pp = data["preprocessing_pipeline"]
        kwargs_pp['save'] = save
        kwargs_pp['dir_out'] = dir_out
        kwargs_pp['tag'] = pm.riderbike_model

        pm.pp_pipeline = PreprocessingPipeline.from_parameters(
            pm.feature_set,
            pm.features,
            **kwargs_pp
        )

        # Configure GMM
        pm.gmm_, _ = pm._init_gmm_from_params(data["gmm_data"]["means"], 
                                              data["gmm_data"]["covariances"],
                                              data["gmm_data"]["weights"],
                                              random_state)
        
        pm.scores_test_ = data["gmm_data"]["scores_test"]
        pm.scores_val_ = pd.DataFrame(data["gmm_data"]["scores_val"], index=[0])
        pm.k_crossval_ = data["gmm_data"]["k_crossval"]
        pm.n_features_ = data["gmm_data"]["n_features"]
        pm.n_samples_train_ = data["gmm_data"]["n_samples_train"]
        pm.n_samples_test_ = data["gmm_data"]["n_samples_test"]
        pm.hyperparameters_ = pd.DataFrame(dict(cov_type=data["gmm_data"]["covariance_type"],
                                   n_components=data["gmm_data"]["n_components"]), index=[0])
        pm.is_fitted_ = True
        return pm