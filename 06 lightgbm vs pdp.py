
import pandas as pd
import numpy as np

import pygam as pg
import lightgbm as lgb

import matplotlib.pyplot as plt
from sklearn.inspection import plot_partial_dependence as pdp

#load dataset
redwine_url = 'https://raw.githubusercontent.com/ianshan0915/medium-articles/master/data/redwine-quality.csv'
redwine = pd.read_csv(redwine_url)

redwine.describe()
redwine['quality'].value_counts(sort=False)

X=redwine.drop('quality', axis=1)
y=redwine['quality']

lgbm=lgb.LGBMRegressor(
    objective='poisson'
    , metric='poisson'
    , num_boost_round=500
    ,num_leaves=10
    ,max_depth=3
    ,reg_lambda=1e5
    ).fit(X, y)


pdp(lgbm, X, [0,1,2], redwine.columns[0:3], percentiles=(0,1))

pdp(lgbm, X, [(0,1)])
pdp(lgbm, X, [(0,2)])
pdp(lgbm, X, [(0,3)])

