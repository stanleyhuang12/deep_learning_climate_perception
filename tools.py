import matplotlib.pyplot as plt
import numpy as np 
import torch
from torch import nn
import torch.functional as F
import torch.optim as optim
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import ParameterGrid



def reset_model_parameters(model):
    for module in model.modules(): 
        if hasattr(module, 'reset_parameters'):
            module.reset_parameters() 



def train_and_evaluate_model(model, X_train, y_train, X_test, y_test, epochs, lr): 
    
    losses = []
    r2_score_tracker = [] 
    print(f'Batching model...')
    
     
    # train_dataset = TensorDataset(X_train, y_train)
    # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    loss_obj = nn.MSELoss()
    optim_obj = optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        # for batch_num, (X, y) in enumerate(train_loader): 
            model.train()
            y_pred = model(X_train)

            # Compute and track MSE loss
            loss = loss_obj(y_pred, y_train.unsqueeze(-1))
            losses.append(loss.item())

            optim_obj.zero_grad()
            loss.backward()
            optim_obj.step()
            
            if epoch % 100 == 0:  
                model.eval()
                with torch.no_grad():
                    y_pred_per_epoch = model.forward(X_train)
                    
                    # Detach for evaluation
                    y_pred_epoch_np = y_pred_per_epoch.cpu().numpy()
                    y_train_np = y_train.cpu().numpy()
            
                r2_score_vals = r2_score(y_train_np, y_pred_epoch_np)
                r2_score_tracker.append(r2_score_vals)
                print("----------------")
                print(f'Training Epoch {epoch}, Training Loss {loss.item()}')
                print(f'Training R2_score: {r2_score_vals}')

    ### Run model on held-out testing data 
    model.eval() 
    
    with torch.no_grad():
        y_held_out_pred = model(X_test)

    loss_score_held_out = loss_obj(y_held_out_pred, y_test.unsqueeze(-1))

    y_held_out_pred_np = y_held_out_pred.cpu().numpy()
    y_test_np = y_test.cpu().numpy()

    r2_score_held_out = r2_score(y_test_np, y_held_out_pred_np)

    print(f"""Evaluation on held out dataset: 
          MSE loss: {loss_score_held_out.item()}, 
          R2_score: {r2_score_held_out}""")

    return model, r2_score_held_out

def tune_hyperparameters(module, X_train, y_train, X_test, y_test, input_dims, output_dims, hidden_state_options, unit_options, epochs_options, lr_options, train_and_evaluate_model):
    results = []
    best_r2 = float('-inf')
    best_model = None
    best_params = None

    param_grid = ParameterGrid({
        'n_hidden_state': hidden_state_options,
        'n_units': unit_options,
        'epochs': epochs_options,
        'lr': lr_options
    })

    for params in param_grid:
        print(f"Training model with params: {params}")
        model = module(input_dims, params['n_hidden_state'], params['n_units'], output_dims)
        trained_model, r2_score_val = train_and_evaluate_model(
            model, X_train, y_train, X_test, y_test, 
            epochs=params['epochs'], lr=params['lr']
        )
        
        results.append({"model": model.__str__,
                        "epoch": params['epochs'],
                        "learning_rate": params['lr'],
                        "r2_score": r2_score_val})

        if r2_score_val > best_r2:
            best_r2 = r2_score_val
            best_model = trained_model
            best_params = params

    print(f"Best R2: {best_r2} with params: {best_params}")
    return best_model, best_params, results



# NOT USING  

def plot_accuracy_scores(score_list, metric_name, show_plot=True):
    score_list = [
        val.detach().numpy() if isinstance(val, torch.Tensor) else val
        for val in score_list
    ]
    
    iterations = np.arange(0, len(score_list)*10, step=10)
    plt.plot(iterations, score_list, label=metric_name)
    
    if show_plot: 
        plt.show()

def plot_multiple_accuracy_scores(metric_names, metrics_list): 
    """
    Plots multiple accuracy score curves on the same figure.
    
    Args:
        metric_names (List[str]): List of metric labels (e.g., model names)
        metrics_list (List[List[float]]): List of score lists
    """
    plt.figure(figsize=(10, 6))
    
    for name, scores in zip(metric_names, metrics_list): 
        plot_accuracy_scores(scores, name, show_plot=False)
    
    plt.xlabel("Training Epochs")
    plt.ylabel("Accuracy")
    plt.title("Accuracy over Training Epochs")
    plt.legend()
    plt.grid(True)
    plt.show()



class Model(nn.Module): 
    def __init__(self, input_dims, h1, h2, output_dims, batch_norm=False): 
        super().__init__() # Instantiate Module
        
        self.first_layer = nn.Linear(input_dims, h1)
        self.hidden_layer = nn.Linear(h1, h2)
        
        ## Batch if necessary
        self.batch_norm = batch_norm
        
        if self.batch_norm: 
            self.bn1 = nn.BatchNorm1d(h1)
            self.bn2 = nn.BatchNorm1d(h2)        
            
        self.output_layer = nn.Linear(h2, output_dims)
    
    def forward(self, x): 
        # We use ReLU activation functions         
        x = self.first_layer(x)
        if self.batch_norm: 
            x = self.bn1(x)   
        x = F.relu(x)
    

        x = self.hidden_layer(x)
        if self.batch_norm: 
            x = self.bn2(x)
            
        x = F.relu(x)
        
        x = self.output_layer(x)
        return x

