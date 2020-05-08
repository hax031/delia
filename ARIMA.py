#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from pandas import DataFrame
from pandas import concat
import pmdarima as pm


# # ARIMA Prediction

# In[26]:


#autoarima model
def arimamodel(timeseries):
    automodel = pm.auto_arima(timeseries, 
                              start_p=1, 
                              start_q=1,
                              test="adf",
                              seasonal=False,
                              trace=True)
    return automodel


# In[27]:


#plot the plot
def plotarima(n_periods, timeseries, automodel):
    # Forecast
    fc, confint = automodel.predict(n_periods=n_periods, return_conf_int=True)
    # Weekly index
    fc_ind = pd.date_range(timeseries.index[timeseries.shape[0]-1], periods=n_periods, freq="Q")
    # Forecast series
    fc_series = pd.Series(fc, index=fc_ind)
    # Upper and lower confidence bounds
    lower_series = pd.Series(confint[:, 0], index=fc_ind)
    upper_series = pd.Series(confint[:, 1], index=fc_ind)
    # Create plot
    plt.figure(figsize=(10, 6))
    plt.plot(timeseries)
    plt.plot(fc_series, color="red")
    plt.xlabel("date")
    plt.ylabel("count")
    plt.fill_between(lower_series.index, 
                     lower_series, 
                     upper_series, 
                     color="k", alpha=.25)
    plt.legend(("past", "forecast", "95% confidence interval"), loc="upper left")
    plt.show()


# In[326]:


ts = Input_c[Input.date <= '2012-01-01']
ts = ts.set_index('date')
ts


# In[327]:


plt.plot(ts)


# In[329]:


plt.plot(ts.diff().diff())
plt.ylabel("the number of patents")


# In[311]:


from numpy import log
#lts = (ts['count'])**(0.5)
lts = log(ts)
plt.plot(lts.diff())
plt.ylabel("the number of patents")


# In[330]:


automodel = arimamodel(ts)
automodel.summary()


# In[331]:


plotarima(10, ts, automodel)


# In[332]:


import math  
y_true = Input['count'][Input.date >'2012-01-01'].values
#y_pred = np.exp(automodel.predict(n_periods=10, return_conf_int=False))-1
y_pred = automodel.predict(n_periods=10, return_conf_int=False)
#y_pred = (automodel.predict(n_periods=10, return_conf_int=False))**2
SSE = ((y_true - y_pred)**2).sum()
SSR = ((y_true - np.mean(ts.values))**2).sum()
OSR2 = 1-SSE/SSR
RMSE = math.sqrt(np.mean((y_true - y_pred)**2))
MAE = np.mean(abs((y_true - y_pred)))
print("OSR2: ", OSR2)
print("MAE: ", MAE)
print("RMSE: ", RMSE)


# In[ ]:




