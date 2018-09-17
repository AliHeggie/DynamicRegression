import pdb
import numpy as np
from scipy.signal import lfilter
import statsmodels.api as sm
import pandas as pd


# Construct the model
class SDR(sm.tsa.statespace.MLEModel):
    def __init__(self, endog, exog, freq_seasonal,level=False,
        exog_errors=False):
        self.level = level
        self.exog_errors = exog_errors
        self.freq_seasonal_periods = [d['period'] for d in freq_seasonal]
        self.freq_seasonal_harmonics = [d.get(
            'harmonics', int(np.floor(d['period'] / 2))) for
            d in freq_seasonal]
        #Get k_exog from size of exog data
        try:
            self.k_exog = exog.shape[1]
        except IndexError:
            exog = np.expand_dims(exog,axis=1)
            self.k_exog = exog.shape[1]
        k_states = (self.level + self.k_exog + sum(self.freq_seasonal_harmonics)*2+
            self.exog_errors*self.k_exog)
        self.k_state_cov = (self.level + self.k_exog * 2 + 
            self.exog_errors*self.k_exog)
        # Initialize the state space model
        super(SDR, self).__init__(endog, k_states=k_states, exog=exog,
         k_posdef=k_states, initialization='approximate_diffuse')
        self.k_posdef=self.k_states

        #Construct the design matrix
        if self.level:
            design = np.vstack((np.ones(self.nobs),self.exog.transpose()))
        else:
            design = self.exog.transpose()
        if self.exog_errors:
            design  = np.vstack((design,self.exog.transpose()))
        for ix, h in enumerate(self.freq_seasonal_harmonics):
            series = self.exog[:,ix]
            lines=np.array([series,np.repeat(0,self.nobs)])
            array=np.vstack(tuple(lines for i in range(0,h)))
            design = np.vstack((design,array))
        self.ssm.shapes['design'] = (self.k_endog,self.k_states,self.nobs)
        self.ssm['design'] = np.expand_dims(design,axis=0)

        #construct transition matrix
        self.transition = np.identity(self.k_states)
        if self.exog_errors:
            i = self.k_exog + self.level
            self.transition[i:i+self.k_exog,i:i+self.k_exog] = 0
        i=self.k_exog + self.level + self.k_exog*self.exog_errors
        for ix, h in enumerate(self.freq_seasonal_harmonics):
            n = 2 * h
            p = self.freq_seasonal_periods[ix]
            lambda_p = 2 * np.pi / float(p)
            t = 0 # frequency transition matrix offset
            for block in range(1, h + 1):
                cos_lambda_block = np.cos(lambda_p * block)
                sin_lambda_block = np.sin(lambda_p * block)
                trans = np.array([[cos_lambda_block, sin_lambda_block],
                                  [-sin_lambda_block, cos_lambda_block]])
                trans_s = np.s_[i + t:i + t + 2]
                self.transition[trans_s, trans_s] = trans
                t += 2
            i += n
        self.ssm['transition']=self.transition

        #construct selection matrix
        self.selection = np.identity(self.k_states)
        self.ssm['selection'] = self.selection
        #construct intercept matrices
        self.obs_intecept = np.zeros(1)
        self.ssm['obs_intercept'] = self.obs_intecept
        self.state_intercept = np.zeros((self.k_states,1))
        self.ssm['state_intercept'] = self.state_intercept
        #construct covariance matrices
        self.obs_cov =  np.zeros(1)
        self.ssm['obs_cov'] = self.obs_cov
        self.state_cov = np.zeros((self.k_states,self.k_states))
        self.ssm['state_cov'] = self.state_cov
        #variance repetitions
        self._var_repetitions = np.ones(self.k_state_cov, dtype=np.int)
        for ix, num_harmonics in enumerate(self.freq_seasonal_harmonics):
            repeat_times = 2 * num_harmonics
            self._var_repetitions[self.k_exog + self.k_exog*self.exog_errors+
                self.level+ix] = repeat_times


    @property
    def start_params(self):
        params = np.zeros(1 + self.level + self.k_exog*2+self.exog_errors*self.k_exog)
        params[0] = np.nanvar(self.ssm.endog)
        if self.level:
            params[1] = np.nanvar(self.ssm.endog)
        return params

    # Describe how parameters enter the model
    def update(self, params, transformed=True, **kwargs):
        params = super(SDR, self).update(params, **kwargs)
        offset = 0
        # Observation covariance
        self.ssm['obs_cov', 0, 0] = params[offset]
        offset += 1
        # State covariance
        variances = params[offset:offset+self.k_states]
        variances = np.repeat(variances, self._var_repetitions)
        self.ssm['state_cov',range(self.k_states),range(self.k_states),0
            ] = variances

    def transform_params(self, unconstrained):
        unconstrained = np.array(unconstrained, ndmin=1)
        constrained = unconstrained**2
        return constrained
    
    def untransform_params(self, constrained):
        constrained = np.array(constrained, ndmin=1)
        unconstrained = np.sqrt(constrained)
        return unconstrained

def plot(results,which="smoothed",alpha=None,figsize=None,combine=False):
    level = results.model.level
    exog_errors = results.model.exog_errors
    if which == "filtered":
        state = results.filtered_state
        fitted_values = results.filter_results.forecasts[0]
    else:
        state = results.smoothed_state
        fitted_values = results.smoother_results.smoothed_forecasts[0]
    endog = results.model.endog

    pd.DataFrame({"endog":endog[:,0],"fitted_values":fitted_values}
        ).plot(figsize=figsize)

    k_exog = results.model.k_exog
    nobs = results.model.nobs
    freq_seasonal_harmonics = results.model.freq_seasonal_harmonics
    nobs = results.model.nobs
    if level:
        level_values = state[0,:]
        regression_coefs = state[range(0+level,k_exog+level),:]
        if combine == True:
            regression_coefs = regression_coefs.sum(axis=0)
            coefs_level = np.vstack((level_values,regression_coefs))
            pd.DataFrame(coefs_level.transpose(),columns=
            ["level","regression_coefs"]
            ).plot(figsize=figsize,subplots=False)
        else:
            coefs_level = np.vstack((level_values,regression_coefs))
            columns = ["level"]+["regression_coef{!r}".format(ix) for ix in 
                range(k_exog)]
            pd.DataFrame(coefs_level.transpose(),columns=columns
                ).plot(figsize=figsize,subplots=True)
    else:
        regression_coefs = state[range(0,k_exog),:]

        if combine == True:
            regression_coefs = regression_coefs.sum(axis=0)
            pd.DataFrame(regression_coefs,columns=
            ["regression_coefs"]
            ).plot(figsize=figsize,subplots=False)
        else:
            pd.DataFrame(regression_coefs.transpose(),columns=
            ["regression_coef{!r}".format(ix) for ix in range(k_exog)]
            ).plot(figsize=figsize,subplots=True)
        if exog_errors:
            state_errors = state[range(k_exog,2*k_exog),:]

            pd.DataFrame(state_errors.transpose(),columns=
            ["state_errors{!r}".format(ix) for ix in range(k_exog)]
            ).plot(figsize=figsize,subplots=True)


    seasonal_regression_coefs = {}
    offset=results.model.k_exog + exog_errors*results.model.k_exog + level
    for ix, harmonics in enumerate(freq_seasonal_harmonics):
        h_idx = np.array([i*2 for i in range(0,harmonics)])+offset
        offset += harmonics*2
        seasonal_regression_coefs['seasonal_regression_coef{!r}'.format(ix)] = state[h_idx,:].sum(axis=0)
    if combine == True:
        pd.DataFrame(seasonal_regression_coefs
        ).sum(axis=1).plot(figsize=figsize,subplots=False)
    else:
        pd.DataFrame(seasonal_regression_coefs
        ).plot(figsize=figsize,subplots=True)