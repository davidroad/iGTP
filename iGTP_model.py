import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.distributions import kl_divergence as kl
from iGTP_Linear import iGTPLinear
from adjustText import adjust_text
import matplotlib.pyplot as plt

class Encoder (nn.Module):
    def __init__(self,args,**kwargs):
        super(Encoder, self).__init__()
        self.args=args
        assert self.args['encoder_normal'] in ['batch','layer'], "only support 'batch' and 'layer' normalization"
        layer_hidden=self.args['encoder_layer_list']
        encoder_list=[]
        for i in range(len(layer_hidden)):
            try:
                encoder_list.append(nn.Linear(layer_hidden[i],layer_hidden[i]))
                encoder_list.append(nn.LazyBatchNorm1d(affine=True))
                encoder_list.append(nn.PReLU())
                encoder_list.append(nn.Dropout(self.args['drop_out']))
                encoder_list.append(nn.Linear(layer_hidden[i],layer_hidden[i+1]))
                encoder_list.append(nn.LazyBatchNorm1d(affine=True))
                encoder_list.append(nn.PReLU())
                encoder_list.append(nn.Dropout(self.args['drop_out']))
            except:
                continue
        self.encoder_module_list=nn.ModuleList(encoder_list)     
            
    def forward(self, x):
        for layer in self.encoder_module_list:
            x = layer(x)
        return x



class iGTP(nn.Module):
    def __init__(self, args, mask_list_dict, **kwargs):
        super(iGTP, self).__init__()
        self.args = args
        self.TP_Gene_mask = mask_list_dict['TP_Gene_mask']
        self.Gene_Gene_mask = mask_list_dict['Gene_Gene_mask']
        
        # Encoder
        self.encoder = Encoder(self.args)
        
        # Mean and logvar layers for variational inference
        self.mean = nn.Sequential(
            nn.LazyLinear(self.args['n_TP']), 
            nn.Dropout(self.args['drop_out'])
        )
        self.logvar = nn.Sequential(
            nn.LazyLinear(self.args['n_TP']), 
            nn.Dropout(self.args['drop_out'])
        )
        
        # Decoder layers
        self.decoder_TP_Gene = nn.Sequential(
            iGTPLinear(self.args, self.TP_Gene_mask, bias=True),
            nn.ReLU()
        )
        self.decoder_Gene_Gene = nn.Sequential(
            iGTPLinear(self.args, self.Gene_Gene_mask, bias=True)
        )
    
    def encode(self, X):
        y = self.encoder(X)
        mu, logvar = self.mean(y), self.logvar(y)
        std = torch.exp(logvar) + self.args['eps']
        
        # Enable anomaly detection
        torch.autograd.set_detect_anomaly(True)
        
        # Identify and fix invalid values in the scale tensor
        min_positive_value = 1e-8
        invalid_mask = std <= 0.0
        std_fixed = std.clone()
        std_fixed[invalid_mask] = min_positive_value
        
        p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std_fixed))
        q = torch.distributions.Normal(mu, std_fixed.sqrt())
        z = q.rsample((self.args['z_sample'],))
        return z, p, q
    
    def encode_z(self, X):
        z, p, q = self.encode(X)
        representation = q.loc.cpu().detach().numpy()
        return representation
    
    def encode_TP(self, X):
        z, p, q = self.encode(X)
        representation = q.loc
        representation_TP = representation.cpu().detach().numpy()
        return representation_TP
    
    def encode_ppi(self, X):
        z, p, q = self.encode(X)
        representation = q.loc.unsqueeze(0)
        representation_PPI = self.decoder_TP_Gene(representation).squeeze()
        representation_PPI = representation_PPI.cpu().detach().numpy()
        return representation_PPI
    
    def forward(self, X):
        library = torch.log(X.sum(1)).unsqueeze(1)
        library = library.unsqueeze(0).expand((self.args['z_sample'], library.size(0), library.size(1)))
        
        z, p, q = self.encode(X)
        
        TP_Gene = self.decoder_TP_Gene(z)
        X_rec = self.decoder_Gene_Gene(TP_Gene)
        
        return_dict = {
            'X_rec': X_rec,
            'p': p,
            'q': q,
            'X': X,
            'z': z
        }
        return return_dict
    
    def compute_pair_bayes_factor(self,array,cell_inf,condiction_column,cell_type_column,stimulate_condiciton,control_condiciton,tsne,use_permutation=True, bays_eps=1e-8 ,m_permutation=5000):
        
        cell_inf['cell_index']=range(len(cell_inf))
        cell_inf_copy = plaid_weights(cell_inf,condiction_column,cell_type_column, tsne)

        control_index=cell_inf[cell_inf[condiction_column]==control_condiciton]['cell_index'].values
        stimulate_index=cell_inf[cell_inf[condiction_column]==stimulate_condiciton]['cell_index'].values
        
        c_weight = cell_inf_copy.iloc[control_index]['weight'].tolist()
        s_weight = cell_inf_copy.iloc[stimulate_index]['weight'].tolist()
        
        c_weight_prob = normalize(c_weight)
        s_weight_prob = normalize(s_weight)
        
        z_tp_c=array[control_index,:]  
        z_tp_s=array[stimulate_index,:]
        
        control_scales, stimulate_scales = pairs_sampler(z_tp_c,z_tp_s,use_permutation=use_permutation,m_permutation=m_permutation,weights1 = c_weight_prob,weights2= s_weight_prob)
        px_control_scales = control_scales.mean(axis=0)
        px_stimulate_scales = stimulate_scales.mean(axis=0)

        mad = np.abs(np.mean(control_scales - stimulate_scales, axis=0))
    
        proba_m1 = np.mean(stimulate_scales > control_scales, axis=0)
        proba_m2 = 1.0 - proba_m1
    
        result = {
            "proba_m1": proba_m1,
            "proba_m2": proba_m2,
            "bayes_factor": np.log(proba_m1 + bays_eps) - np.log(proba_m2 + bays_eps),
            "scale1": px_control_scales,
            "scale2": px_stimulate_scales,
            "mad": mad}
        
        return result

    
    def vae_loss(self, return_dict, epoch=None, anneal_start=None, anneal_time=None):
        beta = self.args['beta']
        
        kld = kl(return_dict['p'], return_dict['q']).sum(dim=-1)
        rec = F.mse_loss(return_dict['X_rec'].squeeze(), return_dict['X'].squeeze(), reduction="sum") 
        loss = torch.mean(rec + beta * kld)
    
        if anneal_start is not None and anneal_time is not None and epoch is not None and epoch > anneal_start:
            beta = min((beta + 1./anneal_time), 1.)
        return loss

    def train_model(self, train_loader, test_loader=None):
        epoch_hist = {'train_loss': [], 'valid_loss': []}
        optimizer = optim.Adam(self.parameters(), lr=self.args['learning_rate'], 
                            weight_decay=self.args['learning_rate_weight_decay'])
    
        train_ES = EarlyStopping(patiences=self.args['train_patience'], verbose=True, mode='train', delta=0.01)
        if test_loader:
            valid_ES = EarlyStopping(patiences=self.args['test_patience'], verbose=True, mode='valid', delta=0.01)
        
        clipper = WeightClipper(frequency=1)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
        
        for epoch in range(self.args['n_epochs']):
            loss_value = 0
            self.train()
            for x_train in train_loader:
                x_train = x_train.to(self.args['device'])
                optimizer.zero_grad()
                return_dict = self.forward(x_train.float())
                loss = self.vae_loss(return_dict, epoch, anneal_start=self.args['anneal_start'], 
                                    anneal_time=self.args['anneal_time'])
                loss_value += loss.item()
                loss.backward()
                nn.utils.clip_grad_norm_(self.parameters(), 1.0)
                optimizer.step()
    
                if self.args['init_type'] in ['pos_normal', 'pos_uniform']:
                    self.decoder_TP_Gene.apply(clipper)
                    self.decoder_Gene_Gene.apply(clipper)
            
            scheduler.step()
            
            # Calculate and log epoch loss
            epoch_loss = self.calculate_epoch_loss(loss_value, train_loader)
            epoch_hist['train_loss'].append(epoch_loss)
            train_ES(epoch_loss)
            
            # Evaluation
            if test_loader:
                test_loss = self.evaluate(test_loader)
                epoch_hist['valid_loss'].append(test_loss)
                valid_ES(test_loss)
                print(f'[Epoch {epoch+1}] | loss: {epoch_loss:.3f} | test_loss: {test_loss:.3f} |', flush=True)
                if valid_ES.early_stop or train_ES.early_stop:
                    print(f'[Epoch {epoch+1}] Early stopping', flush=True)
                    break
            else:
                print(f'[Epoch {epoch+1}] | loss: {epoch_loss:.3f} |', flush=True)
                if train_ES.early_stop:
                    print(f'[Epoch {epoch+1}] Early stopping', flush=True)
                    break

        return epoch_hist
        
    def save_model(self, save_path):
        print('Saving model to ...', save_path)
        torch.save(self.state_dict(), save_path)
    
    def test_model(self, loader):
        """Test model on input loader."""
        test_dict = {}
        loss = 0
        self.eval()
        with torch.no_grad():
            for data in loader:
                data = data.to(self.args['device'])
                return_dict = self.forward(data.float())
                loss += self.vae_loss(return_dict).item()
        
        epoch_loss = self.calculate_epoch_loss(loss, loader)
        test_dict['loss'] = epoch_loss
        return test_dict
    
    def calculate_epoch_loss(self, loss_value, loader):
        if self.args['drop_last_batch']:
            return loss_value / (len(loader) * loader.batch_size)
        else:
            return loss_value / ((len(loader) + 1) * loader.batch_size)
    
    def evaluate(self, test_loader):
        self.eval()
        test_dict = self.test_model(test_loader)
        return test_dict['loss']

# Helper classes (not part of iGTP class, but used by it)
class WeightClipper:
    def __init__(self, frequency=5):
        self.frequency = frequency

    def __call__(self, module):
        if hasattr(module, 'weight'):
            w = module.weight.data
            w = w.clamp(0)
            module.weight.data = w

class EarlyStopping:
    """
    Early stops the training if validation loss doesn't improve after a given patience.
    """
    def __init__(self, patiences=7, verbose=False, delta=0, mode='train'):
        """
        Initialize the EarlyStopping object.

        Args:
            patiences (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                        Default: 0
            mode (str): 'train' or 'valid' mode for early stopping.
                        Default: 'train'
        """
        self.patience = patiences
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.mode = mode

    def __call__(self, val_loss):
        """
        Check if the training should be stopped based on the validation loss.

        Args:
            val_loss (float): The current validation loss.

        Returns:
            None
        """
        score = -val_loss

        if self.best_score is None:
            # First iteration
            self.best_score = score
            # self.save_checkpoint(val_loss, model)  # Commented out, not implemented
        elif (self.mode == 'valid' and score <= self.best_score + self.delta) or \
            (self.mode == 'train' and score <= self.best_score + self.delta):
            # Validation loss hasn't improved
            self.counter += 1
            # print(f'EarlyStopping counter: {self.counter} out of {self.patience}')  # Commented out
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            # Validation loss has improved
            self.best_score = score
            # self.save_checkpoint(val_loss, model)  # Commented out, not implemented
            self.counter = 0

def plaid_weights(cell_inf, condition_column, cell_type_column, tsne):
    """
    Calculate weights for cells based on their distribution in t-SNE space.

    Args:
        cell_inf (pd.DataFrame): DataFrame containing cell information.
        condition_column (str): Name of the column containing condition information.
        cell_type_column (str): Name of the column containing cell type information.
        tsne (np.array): t-SNE embeddings for the cells.

    Returns:
        pd.DataFrame: A copy of cell_inf with an additional 'weight' column.
    """
    # Create a copy of the input DataFrame to avoid modifying the original
    cell_inf_copy = cell_inf.copy()
    
    # Initialize the weight column with zeros
    cell_inf_copy['weight'] = 0

    # Iterate over each unique cell type and condition
    for celltype in cell_inf[cell_type_column].unique():
        for arm in cell_inf[condition_column].unique():
            # Get indices of cells matching the current cell type and condition
            index = cell_inf[
                (cell_inf[condition_column] == arm) & 
                (cell_inf[cell_type_column] == celltype)
            ]['cell_index'].values

            # Extract t-SNE coordinates for the current subset of cells
            z_type = tsne[index, :]
            if z_type.shape[0] > 0:
                # Calculate the bounding box of the t-SNE coordinates
                z_min, z_max = z_type.min(axis=0), z_type.max(axis=0)
                z_span = (z_max - z_min)

                # Calculate the volume of the bounding box
                volume = np.prod(z_span)

            else:
                volume = 0.0

            # Assign the calculated volume as weight to the corresponding cells
            cell_inf_copy.loc[
                (cell_inf_copy[condition_column] == arm) & 
                (cell_inf_copy[cell_type_column] == celltype), 
                'weight'
            ] = volume

    return cell_inf_copy

def normalize(weight):
    """
    Normalize a list of weights by dividing each weight by the sum of all weights.

    Args:
        weight (list): A list of numerical weights.

    Returns:
        list: A list of normalized weights.
    """
    sum_of_numbers = sum(weight)

    # Normalize the numbers by dividing each by the sum
    normalized_numbers = [x / sum_of_numbers for x in weight]
    
    # Uncomment for debugging:
    # print(normalized_numbers)
    
    return normalized_numbers

def pairs_sampler(
    arr1,
    arr2,
    use_permutation: bool = True,
    m_permutation: int = None,
    sanity_check_perm: bool = False,
    weights1=None,
    weights2=None,
) -> tuple:
    """
    Creates more pairs for estimating a double sum.

    This function virtually increases the number of samples by considering more pairs
    to better estimate a double summation operation.

    Args:
        arr1: Samples from population 1 (np.ndarray or similar).
        arr2: Samples from population 2 (np.ndarray or similar).
        use_permutation (bool): Whether to mix samples from both populations.
        m_permutation (int): Number of permutations to generate.
        sanity_check_perm (bool): If True, resulting mixed arrays arr1 and arr2 are mixed together.
        weights1: Probabilities associated with array 1 for random sampling.
        weights2: Probabilities associated with array 2 for random sampling.

    Returns:
        tuple: (new_arr1, new_arr2)
    """
    if use_permutation:
        n_arr1 = arr1.shape[0]
        n_arr2 = arr2.shape[0]
        
        if not sanity_check_perm:
            # Case 1: No permutation, sample from A and then from B
            u = np.random.choice(n_arr1, size=m_permutation, p=weights1)
            v = np.random.choice(n_arr2, size=m_permutation, p=weights2)
            first_set = arr1[u]
            second_set = arr2[v]
        else:
            # Case 2: Permutation, sample from A+B twice (sanity check)
            concat_arr = np.concatenate((arr1, arr2))
            total_size = n_arr1 + n_arr2
            u = np.random.choice(total_size, size=m_permutation, p=weights1)
            v = np.random.choice(total_size, size=m_permutation, p=weights2)
            first_set = concat_arr[u]
            second_set = concat_arr[v]
    else:
        first_set = arr1
        second_set = arr2
    
    return first_set, second_set

def plot_dfe(dfe_res,
            pathway_list,
            sig_lvl,
            lfc_lvl,
            to_plot=None,
            metric='mad',
            figsize=[12,5],
            s=10,
            fontsize=10,
            textsize=8,
            title=False,
            save=False):
    """
    Plot Differential Factor Expression results.

    Args:
        dfe_res (dict): Dictionary with differential factor analysis results from VAE.
        pathway_list (list): List with names of factors.
        sig_lvl (float): log_e(BF) threshold for significance.
        lfc_lvl (float): Threshold for metric significance level (MAD or LFC).
        to_plot (dict): Subset of factors to annotate on plot. If None, defaults to all significant.
        metric (str): Metric to use for Y-axis. 'mad' or 'lfc' (default: 'mad').
        figsize (tuple): Figure size.
        s (int): Size of dots for scatter plot.
        fontsize (int): Size of text on axes.
        textsize (int): Size of text for annotation on plot.
        title (str): Plot title (optional).
        save (str): Path to save figure (optional).
    """
    # Initialize plot
    plt.figure(figsize=figsize)
    xlim_v = np.abs(dfe_res['bayes_factor']).max() + 0.5
    ylim_v = dfe_res[metric].max() + 0.5

    # Identify significant points
    idx_sig = np.arange(len(dfe_res['bayes_factor']))[
        (np.abs(dfe_res['bayes_factor']) > sig_lvl) & 
        (np.abs(dfe_res[metric]) > lfc_lvl)
    ]

    # Plot points
    plt.scatter(dfe_res['bayes_factor'], dfe_res[metric], color='darkgrey', s=s, alpha=0.8, linewidth=0)
    plt.scatter(dfe_res['bayes_factor'][idx_sig], dfe_res[metric][idx_sig], 
                marker='*', color='salmon', s=s*2, linewidth=0)

    # Add threshold lines
    plt.vlines(x=[-sig_lvl, sig_lvl], ymin=-0.5, ymax=ylim_v, color='darkgreen', 
            linestyles='--', linewidth=2., alpha=0.1)
    plt.hlines(y=lfc_lvl, xmin=-xlim_v, xmax=xlim_v, color='darkcyan', 
            linestyles='--', linewidth=2., alpha=0.1)

    # Add text annotations
    texts = []
    if to_plot is None:
        for i in idx_sig:
            name = pathway_list[i]
            x, y = dfe_res['bayes_factor'][i], dfe_res[metric][i]
            texts.append(plt.text(x=x, y=y, s=name, fontdict={'size': textsize}))
    else:
        idx_plot = [(pathway_list.index(f), to_plot[f]) for f in to_plot]
        for i, name in idx_plot:
            x, y = dfe_res['bayes_factor'][i], dfe_res[metric][i]
            texts.append(plt.text(x=x, y=y, s=name, fontdict={'size': textsize}))

    # Set labels and title
    plt.xlabel(r'$\log_e$(Bayes factor)', fontsize=fontsize)
    plt.ylabel('MAD' if metric == 'mad' else r'$\log{|(FC_{z})|}$', fontsize=fontsize)
    plt.ylim([0, ylim_v])
    plt.xlim([-xlim_v, xlim_v])
    if title:
        plt.title(f"{title}(|K|>{sig_lvl:.1f})")

    # Adjust text positions
    adjust_text(texts, only_move={'texts': 'xy'}, 
                arrowprops=dict(arrowstyle="-", color='k', lw=2))

    # Set tick font sizes
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)

    # Save figure if requested
    if save:
        plt.savefig(save, format='pdf', dpi=150, bbox_inches='tight')

    plt.show()
