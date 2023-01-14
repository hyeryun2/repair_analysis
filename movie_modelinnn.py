#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
import numpy as np
import scipy as sp
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


# In[8]:


data = pd.read_csv("documents/movie.csv", index_col=0)


# In[ ]:


dd_set = pd.read_csv("data/value_data.csv", index_col=0)


# In[ ]:


from sklearn.cross_validation import train_test_split
X = d_df.ix[:,:-1]
y = d_df.ix[:,-1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)


# In[ ]:


X = modeling_data.ix[:,:-1]
scaler = StandardScaler(with_mean=False)
X_scaled = scaler.fit_transform(X)

dfX0 = pd.DataFrame(X_scaled, columns=X.columns)
dfX = sm.add_constant(dfX0)
dfy = pd.DataFrame(modeling_data.ix[:,-1], columns=["audience"])
d_df = pd.concat([dfX, dfy], axis=1)
d_df.head()


# In[ ]:




