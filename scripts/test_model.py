import pandas as pd
from sktime.classification.shapelet_based import ShapeletTransformClassifier
from sktime.classification.distance_based import KNeighborsTimeSeriesClassifier
from sktime.classification.interval_based import TimeSeriesForestClassifier
from sklearn.ensemble import RandomForestClassifier
import pickle

test = pd.read_csv('/home/illania/project/datasets/data_test.csv')

y_test = test['y'].astype('int').values
X_test = test.drop('y', axis=1).values

model_1 = ShapeletTransformClassifier(estimator=RandomForestClassifier(n_estimators=100),
                                  n_shapelet_samples=100,
                                  max_shapelets=100,
                                  batch_size=20)
                                  
model_2 = TimeSeriesForestClassifier(n_estimators=100,random_state=47)

model_3 = KNeighborsTimeSeriesClassifier(n_neighbors=1, distance="ddtw")
                                  
with open('/home/illania/project/models/model_1.pickle', 'rb') as f:
    model_1 = pickle.load(f)
    
with open('/home/illania/project/models/model_2.pickle', 'rb') as f:
    model_2 = pickle.load(f)
    
with open('/home/illania/project/models/model_3.pickle', 'rb') as f:
    model_3 = pickle.load(f)

test_score = []

test_score.append(model_1.score(X_test, y_test))
test_score.append(model_2.score(X_test, y_test))
test_score.append(model_3.score(X_test, y_test))

models = ['ShapeletTransformClassifier','TimeSeriesForestClassifier','KNeighborsTimeSeriesClassifier']

results = list(zip(models,test_score))
results_df = pd.DataFrame(results, columns=['model','test_score'])

results_df.to_csv('/home/illania/project/results.csv')
