from sklearn.model_selection import StratifiedKFold, KFold
import torch
import numpy as np
import pickle5 as pickle

class KFoldTorch:
    """
    A class to perform K-Fold cross-validation for PyTorch models.
    """

    def __init__(self, args):
        """
        Initialize the KFoldTorch object.

        Args:
            args (dict): A dictionary containing configuration parameters.
        """
        self.args = args
        self.cv_res_dict = {c: {} for c in range(self.args['cv_fold'])}

        # Set up GPU if available and requested
        if torch.cuda.is_available() and self.args['using_gpu']:
            num_gpus = torch.cuda.device_count()
            gpu_number = self.args['device_nu']
            self.args['device'] = torch.device(f"cuda:{gpu_number}")
        else:
            self.args['device'] = torch.device("cpu")
        
        print(f"Model(s) will be saved at {self.args['model_dir']} using {self.args['model_prefix']} as prefix", flush=True)
    
    def train_kfold(self, MyModel, mask_list_dict, dataset):
        """
        Run K-Fold cross-validation training.

        Args:
            MyModel (class): The model class to be instantiated and trained.
            mask_list_dict (dict): A dictionary of masks used by the model.
            dataset (torch.utils.data.Dataset): The dataset to be split and used for training.

        Returns:
            None
        """
        kfold = KFold(n_splits=self.args['cv_fold'], shuffle=True)
        best_val_loss = float('inf')
        best_cv = 0
        best_model = None

        for i, (train_idx, test_idx) in enumerate(kfold.split(dataset)):
            # Initialize model for this fold
            model = MyModel(self.args, mask_list_dict)
            model.to(self.args['device'])
            
            print(f'Training fold {i}', flush=True)

            # Prepare data loaders
            train_ds = torch.utils.data.Subset(dataset, train_idx)
            test_ds = torch.utils.data.Subset(dataset, test_idx)
            
            train_loader = torch.utils.data.DataLoader(
                train_ds, 
                batch_size=self.args['batch_size'], 
                shuffle=True,
                drop_last=self.args['drop_last_batch'], 
                num_workers=self.args['data_num_workers']
            )
            
            test_loader = torch.utils.data.DataLoader(
                test_ds, 
                batch_size=self.args['batch_size'], 
                shuffle=True,
                drop_last=self.args['drop_last_batch'], 
                num_workers=self.args['data_num_workers']
            )

            # Train model
            epoch_hist = model.train_model(train_loader, test_loader=test_loader)

            # Save training history
            self.cv_res_dict[i]['history'] = epoch_hist

            if 'valid_loss' in epoch_hist.keys():
                self.cv_res_dict[i]['best_valid_loss'] = np.min(epoch_hist['valid_loss'])
                if best_val_loss > np.min(epoch_hist['valid_loss']):
                    best_val_loss = np.min(epoch_hist['valid_loss'])
                    best_cv = i
                    best_model = model

            # Save model for this fold
            full_path = f"{self.args['model_dir']}{self.args['model_prefix']}fold_{i}.pt"
            print(f'Saving model at {full_path}', flush=True)
            model.save_model(full_path)

            # Clean up to free memory
            del model
            torch.cuda.empty_cache()
            best_model.to(torch.device("cpu"))

        # Save best model and results
        self.best_cv = best_cv
        print(f'Best Fold: {self.best_cv}', flush=True)
        path_best = f"{self.args['model_dir']}{self.args['model_prefix']}best_fold.pt"
        best_model.save_model(path_best)

        file_path = f"{self.args['model_dir']}{self.args['model_prefix']}all_loss.pickle"
        with open(file_path, 'wb') as file:
            pickle.dump(self.cv_res_dict, file)

        del best_model
