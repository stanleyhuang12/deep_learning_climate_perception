from tools_rec import *
import os
import numpy as np
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
from ray import train
# from ray.train import ScalingConfig
from ray.train.torch import TorchTrainer
from ray import tune
from ray.tune.schedulers import ASHAScheduler # Scheduler 
from torcheval.metrics import R2Score 
from ray.tune.search import hyperopt #Hyperopt performs tree-structured parzen estimator


"=====================Bayesian hyperparameter search====================="
"""
Parameters to search from: 

- Learning rate (policy)
- Batch size
- Drop out rate
- K for sparsification
- Dimensionality of the neural network 
- max epochs 
- Layers to sparsify 

"""


data = pd.read_csv(os.path.join(os.getcwd(), 'data/CCES2012_CSVFormat_NEW.csv'))
# for col in data.columns: 
#     if col.startswith('cd'): 
#         print(col)
X_c = data.drop(['CC12'], axis=1).values.astype(np.float32)
y_c = data['CC12'].values.reshape(-1, 1).astype(np.float32)


## Brute force trial without using Ray tune yet 




search_space_init_configs = {
    'num_layers': tune.randint(1, 11),
    'base_exp': tune.randint(3, 12),
    'lr': tune.loguniform(1e-5, 2e-1)
}


# param_configs = {
#     'num_layers': 3,
#     'base_exp': 6,
#     'lr': 0.000575
# }

train_configs = {
    'file_path': '/Users/stanleyhuang/Desktop/01 Projects/YAB/cds_climate_change_perception/Climate Change Perceptions/data/CCES2012_CSVFormat_NEW.csv',
    'input_dims': 2668,
    'output_dims': 1,
    'max_epochs': 70,
    'batch_size': 12
}

# os.environ["TUNE_DISABLE_STRICT_METRIC_CHECKING"] = "1"


def new_train_fn(config): 
    
    # Train configs is a dictionary of items that does not vary 
    merged_configs = {**train_configs, **config}
    dataset = load_dataset(merged_configs['file_path'])
    
    X = dataset.drop(['CC12'], axis=1).values.astype(np.float32)
    y = dataset['CC12'].values.reshape(-1, 1).astype(np.float32)
        
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    for train_idx, valid_idx in kf.split(X): 
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[valid_idx], y[valid_idx]
    
    # Convert to tensors 
    X_train, y_train = torch.from_numpy(X_train), torch.from_numpy(y_train)
    X_test, y_test = torch.from_numpy(X_test), torch.from_numpy(y_test)
    
    model = initialize_model(merged_configs)
    
    # model = train.torch.prepare_model(model)
    
    criterion = nn.MSELoss()
    metric = R2Score()
    optimizer = optim.Adam(model.parameters(), lr=merged_configs['lr'])
    
    tensor_dataset = TensorDataset(X_train, y_train)
    dataloader = DataLoader(tensor_dataset, batch_size=merged_configs['batch_size'], shuffle=True)
    
    r2_score_tracker = float('-inf')
    for epoch in range(merged_configs['max_epochs']):
        model.train()
        train_loss = 0 
        
        for batch_X, batch_y in dataloader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Compute model R2 
        valid_loss = 0
        model.eval()
        with torch.no_grad(): 
            y_pred = model.forward(X_test)
            valid_loss = criterion(y_pred, y_test)
            valid_loss = valid_loss.item()

            
        averaged_train_loss = train_loss / X_train.size(1) # Average training loss per epoch 
        metric.reset()
        metric.update(y_pred, y_test)
        model_r2 = metric.compute().item()
        
        print(f"Epoch {epoch}: Current R2 = {model_r2}, Best R2 = {r2_score_tracker}")
        
        metrics = {
            "step": epoch,
            "train_loss": averaged_train_loss,
            "valid_loss": valid_loss,
            "r2_score": model_r2
        }
        
        if model_r2 > r2_score_tracker: 
            ## Keep track of best model r2 score and save it as a checkpoint to call or continue from 
            r2_score_tracker = model_r2 
            
            with tempfile.TemporaryDirectory() as tmpdir: 
            ### REVISSE FOR PERSISTENT SˇORAGE PERSISTING CHECKPOINTS 
                checkpoint_to_track = None
                torch.save(
                    model.state_dict(),
                    os.path.join(tmpdir, "model.pth")
                )
                checkpoint_to_track = Checkpoint.from_directory(tmpdir)
                
                train.report(metrics, checkpoint=checkpoint_to_track)
        else: 
            train.report(metrics)
        
        print(f'Epoch {epoch}: Training loss {averaged_train_loss}, Validation loss {valid_loss}, R2 Score {model_r2}')
        
        
    
# trainer = TorchTrainer(new_train_fn, train_loop_config=param_configs, 
#                        run_config=tune.RunConfig('/Users/stanleyhuang/Desktop/01 Projects/YAB/cds_climate_change_perception/Climate Change Perceptions/checkpoints/train_practice',
#                                             checkpoint_config=tune.CheckpointConfig(num_to_keep=1, 
#                                                                                checkpoint_score_attribute='valid_loss'))
# )
 
# res = trainer.fit()


# if there is a checkpoint configuration, should we use this to do something? 
asha_scheduler = ASHAScheduler(max_t=70,
                               grace_period=7,
                               reduction_factor=2, 
                               metric='valid_loss',
                               mode='min')

tree_parzen_estimator = hyperopt.HyperOptSearch(metric='valid_loss', 
                                                mode='min')

tuner = tune.Tuner(trainable=new_train_fn, 
                   param_space=search_space_init_configs, 
                   tune_config=tune.TuneConfig(num_samples=4,
                                          max_concurrent_trials=4, 
                                          scheduler=asha_scheduler, 
                                          search_alg=tree_parzen_estimator),
                   run_config=tune.RunConfig(name='env_pred_tpe_algo_run',
                                        storage_path='/Users/stanleyhuang/Desktop/01 Projects/YAB/cds_climate_change_perception/Climate Change Perceptions/checkpoints',
                                        checkpoint_config=tune.CheckpointConfig(num_to_keep=10,
                                        checkpoint_score_attribute='valid_loss')
                                        ))

results = tuner.fit()

