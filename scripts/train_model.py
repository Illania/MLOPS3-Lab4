import pandas as pd
import numpy as np
from sktime.classification.shapelet_based import ShapeletTransformClassifier
from sktime.classification.distance_based import KNeighborsTimeSeriesClassifier
from sktime.classification.interval_based import TimeSeriesForestClassifier
from sklearn.ensemble import RandomForestClassifier
import pickle
import os

train = pd.read_csv('/home/illania/project/datasets/data_train.csv')

y_train = train['y'].astype('int').values
X_train = train.drop('y', axis=1).values

model_1 = ShapeletTransformClassifier(estimator=RandomForestClassifier(n_estimators=100),
                                  n_shapelet_samples=100,
                                  max_shapelets=100,
                                  batch_size=20)

model_1.fit(X_train, y_train)

with open('/home/illania/project/models/model_1.pickle', 'wb') as f:
    pickle.dump(model_1, f)
    
    
model_2 = TimeSeriesForestClassifier(n_estimators=100,random_state=47)

model_2.fit(X_train, y_train)

with open('/home/illania/project/models/model_2.pickle', 'wb') as f:
    pickle.dump(model_2, f)
    
    
model_3 = KNeighborsTimeSeriesClassifier(n_neighbors=1, distance="ddtw")

model_3.fit(X_train, y_train)

with open('/home/illania/project/models/model_3.pickle', 'wb') as f:
    pickle.dump(model_2, f)
