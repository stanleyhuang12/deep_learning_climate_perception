from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import pandas as pd 


df = pd.read_csv('CCES2012_CSVFormat_NEW.csv')

X = df.drop(['CC12'], axis=1)
y = df['CC12']

## Baseline with non-standardized features 

ridge = Ridge()
param_grid = {'alpha': [0.1, 1, 10, 100, 200, 300], 'fit_intercept': [True, False]}
grid_search = GridSearchCV(estimator=ridge, param_grid=param_grid, cv=5)
grid_search.fit(X, y)
print("Best parameters for baseline Ridge Regression: ", grid_search.best_params_) 
print("R2 score: ", grid_search.best_score_)

"""
Best parameters for baseline Ridge Regression:  {'alpha': 100}
R2 score:  0.4185151366506159
"""


## Baseline with standardized features 

sc = StandardScaler()
X_c = sc.fit_transform(X)

ridge_2 = Ridge()

grid_search_sc = GridSearchCV(ridge_2, param_grid, cv=5)
grid_search_sc.fit(X_c, y)
print("Best parameters for baseline Ridge Regression with standardized features: ", grid_search_sc.best_params_) 
print("R2 score with standardized features: ", grid_search_sc.best_score_)


model_response = """
Best parameters for baseline Ridge Regression with standardized features:  {'alpha': 200, 'fit_intercept': True}
R2 score with standardized features:  0.4184565576233914
"""
