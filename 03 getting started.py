

# https://codeburst.io/pygam-getting-started-with-generalized-additive-models-in-python-457df5b4705f


import pandas as pd        
from pygam import LogisticGAM
from sklearn.datasets import load_breast_cancer

#load the breast cancer data set
data = load_breast_cancer()

#keep first 6 features only
df = pd.DataFrame(data.data, columns=data.feature_names)[['mean radius', 'mean texture', 'mean perimeter', 'mean area','mean smoothness', 'mean compactness']]
target_df = pd.Series(data.target)

df.describe()



#building simple logistic regression

X = df[['mean radius', 'mean texture', 'mean perimeter', 'mean area','mean smoothness', 'mean compactness']]
y = target_df

#Fit a model with the default parameters
gam = LogisticGAM().fit(X, y)
gam.summary()