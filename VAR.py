#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
from numpy.random import seed
seed(1)
#import tensorflow as tf
#tf.random.set_seed(2)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
#from keras.models import Sequential
#from keras.layers import Dense
#from keras.layers import LSTM
#from keras.layers import Dropout
#from keras.layers import Flatten
from sklearn.metrics import mean_squared_error
#from keras.callbacks import EarlyStopping, ModelCheckpoint


# In[2]:


# frame a sequence as a supervised learning problem
def timeseries_to_supervised(sqs, lag):
    cols, names = list(), list()
    F = sqs.drop(columns={"date"}, axis=1)
    columns = F.columns
    n_var = F.shape[1]
    for i in range(lag, 0, -1):
        cols.append(F.shift(i))
        names +=([(columns[j]+'_t-%d' % i) for j in range(n_var)])
    cols.append(F)
    names += F.columns.tolist()
    reframed = concat(cols, axis=1)
    reframed.columns = names
    reframed = reframed.drop(reframed.columns[[-1,-2]], axis=1)
    return reframed


# In[3]:


def scale(train, test):
    # fit scaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler = scaler.fit(train)
    # transform train
    train_scaled = scaler.transform(train)
    # transform test
    test_scaled = scaler.transform(test)
    return scaler, train_scaled, test_scaled


# In[4]:


# build the lstm model
def fit_lstm(train,time_step, batch_size, neurons):
    mean_loss = 0
    X, y = train[:, 3*(4-time_step):-1], train[:, -1]
    X = X.reshape(X.shape[0], time_step, int(X.shape[1]/time_step))
    model = Sequential()
    model.add(LSTM(units = neurons, batch_input_shape=(batch_size, X.shape[1], X.shape[2]), 
                   stateful=True, return_sequences = True))
    model.add(LSTM(units = neurons, return_sequences = False))
    model.add(Dense(units = 1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    for i in range(30):
        history = model.fit(X, y, epochs=1000, batch_size=batch_size, verbose=0, 
                        shuffle=False, validation_split = 0.2)
        model.reset_states()
        mean_loss += min(history.history['val_loss'])
    return mean_loss/30


# # VAR model

# In[ ]:


def varData(M):
    data = M
    data = data.drop(columns = 'Inventor')
    data['date'] = pd.to_datetime(data['date'])
    data = data[(data['date'] > '1997-01-01') & (data['date'] < '2019-01-01')]
    data.set_index('date', inplace = True)
    data = data.rename(columns={"counts": "patent", "Citation": "citing_cnt"})
    return data


# In[ ]:


def patentCitingPlot(data):
    plt.figure(figsize = (8,5))
    plt.plot(data['patent'],"x-",label="patent")
    plt.plot(data['citing_cnt'],label="citation")
    plt.legend(bbox_to_anchor=(1.0, 1), loc=1, borderaxespad=0.)
    plt.show()


# In[ ]:


# causality test
from statsmodels.tsa.stattools import grangercausalitytests
maxlag=12
test = 'ssr_chi2test'
def grangers_causation_matrix(data, variables, test='ssr_chi2test', verbose=False):    
    """Check Granger Causality of all possible combinations of the Time series.
    The rows are the response variable, columns are predictors. The values in the table 
    are the P-Values. P-Values lesser than the significance level (0.05), implies 
    the Null Hypothesis that the coefficients of the corresponding past values is 
    zero, that is, the X does not cause Y can be rejected.

    data      : pandas dataframe containing the time series variables
    variables : list containing names of the time series variables.
    """
    df = pd.DataFrame(np.zeros((len(variables), len(variables))), columns=variables, index=variables)
    for c in df.columns:
        for r in df.index:
            test_result = grangercausalitytests(data[[r, c]], maxlag=maxlag, verbose=False)
            p_values = [round(test_result[i+1][0][test][1],4) for i in range(maxlag)]
            if verbose: print(f'Y = {r}, X = {c}, P Values = {p_values}')
            min_p_value = np.min(p_values)
            df.loc[r, c] = min_p_value
    df.columns = [var + '_x' for var in variables]
    df.index = [var + '_y' for var in variables]
    return df


# In[ ]:


# stationary test
def adfuller_test(series, signif=0.05, name='', verbose=False):
    """Perform ADFuller to test for Stationarity of given series and print report"""
    r = adfuller(series, autolag='AIC')
    output = {'test_statistic':round(r[0], 4), 'pvalue':round(r[1], 4), 'n_lags':round(r[2], 4), 'n_obs':r[3]}
    p_value = output['pvalue'] 
    def adjust(val, length= 6): return str(val).ljust(length)

    # Print Summary
    print(f'    Augmented Dickey-Fuller Test on "{name}"', "\n   ", '-'*47)
    print(f' Null Hypothesis: Data has unit root. Non-Stationary.')
    print(f' Significance Level    = {signif}')
    print(f' Test Statistic        = {output["test_statistic"]}')
    print(f' No. Lags Chosen       = {output["n_lags"]}')

    for key,val in r[4].items():
        print(f' Critical value {adjust(key)} = {round(val, 3)}')

    if p_value <= signif:
        print(f" => P-Value = {p_value}. Rejecting Null Hypothesis.")
        print(f" => Series is Stationary.")
    else:
        print(f" => P-Value = {p_value}. Weak evidence to reject the Null Hypothesis.")
        print(f" => Series is Non-Stationary.")    


# # fit model

# In[ ]:


def fitVar(input):
    from statsmodels.tsa.api import VAR
    from statsmodels.tsa.stattools import adfuller
    from statsmodels.tools.eval_measures import rmse, aic

    Data = varData(input)

    train = Data[(Data.index < '2016-01-01') ]
    test = Data[(Data.index >= '2016-01-01') ]

    # differenced once
    differenced = train.diff().dropna()
    #differenced = differenced.diff().dropna()

    model = VAR(differenced)

    # find the order
    order = {}
    for i in [1,2,3,4,5,6,7,8,9,10,11,12]:
        result = model.fit(i)
        order[i] = result.aic


    o = min(order,key = order.get)

    model_fitted = model.fit(o)
    model_fitted.summary()

    forecast_input = differenced.values[-o:]

    nobs = len(test)

    fc = model_fitted.forecast(y=forecast_input, steps=nobs)
    forecast = pd.DataFrame(fc, index=Data.index[-nobs:], columns=Data.columns + '_1d')

    forecast

    def invert_transformation(df_train, df_forecast, second_diff=False):
        """Revert back the differencing to get the forecast to original scale."""
        df_fc = df_forecast.copy()
        columns = df_train.columns
        for col in columns:        
            # Roll back 2nd Diff
            if second_diff:
                df_fc[str(col)+'_1d'] = (df_train[col].iloc[-1]-df_train[col].iloc[-2]) + df_fc[str(col)+'_2d'].cumsum()
            # Roll back 1st Diff
            df_fc[str(col)+'_forecast'] = df_train[col].iloc[-1] + df_fc[str(col)+'_1d'].cumsum()
        return df_fc

    results = invert_transformation(train, forecast, second_diff=False) 

    results.loc[:,['patent_forecast', 'citing_cnt_forecast']]

    fig, axes = plt.subplots(nrows=int(len(Data.columns)/2), ncols=2, dpi=150, figsize=(10,4))
    for i, (col,ax) in enumerate(zip(Data.columns, axes.flatten())):
        results[col+'_forecast'].plot(legend=True, ax=ax).autoscale(axis='x',tight=True)
        test[col][-nobs:].plot(legend=True, ax=ax);
        ax.set_title(col + ": Forecast vs Actuals")
        ax.xaxis.set_ticks_position('none')
        ax.yaxis.set_ticks_position('none')
        ax.spines["top"].set_alpha(0)
        ax.tick_params(labelsize=6)

    plt.tight_layout()
    return results.loc[:,['patent_forecast']]


# In[ ]:


quarterlyResult = fitVar(Q)

