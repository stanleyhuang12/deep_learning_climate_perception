import pandas as pd 
from torch import nn
from tools import *
from skorch import NeuralNetRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from skorch.callbacks import EarlyStopping
from skorch.dataset import ValidSplit


## Organize data 

data = pd.read_csv('CCES2012_CSVFormat_NEW.csv')

non_county_cols = [col for col in data.columns if not col.startswith('cd')]

df= data[non_county_cols]

X = df.drop(['CC12'], axis=1).values
y = df['CC12'].values

## Create neural network model 
class DynamicNet(nn.Module): 
    def __init__(self, input_dims, list_n_units, output_dims): 
        self.list_n_units = list_n_units
        
        super().__init__()
        layers = []
        
        layers.append(nn.Linear(input_dims, list_n_units[0]))
        layers.append(nn.ReLU())
        
        for i in range(len(list_n_units) - 1): 
            layers.append(nn.Linear(list_n_units[i], list_n_units[i+1]))
            layers.append(nn.ReLU())
        
        layers.append(nn.Linear(list_n_units[-1], output_dims))
        
        self.neural_network = nn.Sequential(*layers)
    
    def forward(self, x): 
        
        return self.neural_network(x)
    
    def __str__(self): 
        
        str_val = "".join(["_"+str(val) for val in self.list_n_units])
        
        return f"dnn_{len(self.list_n_units)}{str_val}"
    

# General workflow
# Initialize an instance of NN 

nn1 = DynamicNet(input_dims=32, list_n_units=[32, 32, 32], output_dims=1)

# Reset weights and parameters 
reset_model_parameters(nn1)
train_eval_held_out(nn1, X, y, epochs=20, criterion=nn.MSELoss(), lr=0.015, split_num=5)

nn1_wrapper = NeuralNetRegressor(
    module=DynamicNet,
    criterion=nn.MSELoss,
    module__input_dims=32,
    module__list_n_units=[32, 32, 32],
    module__output_dims=1,
    max_epochs=30,
    optimizer=optim.Adam,
    optimizer__lr=0.1,
    callbacks = [EarlyStopping(patience=7)],
    train_split=ValidSplit(0.2) # We set internal train_split to 0 because GridSearch has cv=5
)


wrapper_param_grids = { 
        'module__list_n_units': [[32, 32, 32], [64, 64, 64], [128, 128, 128], [128, 64, 32]],
        'optimizer__lr': [0.00375, 0.01, 0.02, 0.0275]
}


grid_search_cv = GridSearchCV(nn1_wrapper, param_grid=wrapper_param_grids, n_jobs=-1, verbose=1)


grid_search_cv.fit(X.astype(np.float32), y.astype(np.float32).reshape(-1, 1))

print(grid_search_cv.best_params_)
"""Best parameters: 
{'module__list_n_units': [64, 64, 64], 'optimizer__lr': 0.01}
"""
print(grid_search_cv.best_score_)
"""Best score (I believe the default for GridSearch on regression tasks is R2 score)
0.4202298879623413
"""

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

grid_search_cv.fit(X_scaled.astype(np.float32), y.astype(np.float32).reshape(-1, 1))




