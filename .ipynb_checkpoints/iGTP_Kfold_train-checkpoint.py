import os
import argparse
import scanpy as sc
import pickle
import yaml
from iGTP_Linear import *
from iGTP_model import *
from preprocess import *
from learning_utilies import *

def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def load_data(args):
    """Load and preprocess the data."""
    adata = sc.read(args['sc_data_path'])
    print(f"Original data shape: {adata.X.shape}")

    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)
    adata.var['mt'] = adata.var_names.str.startswith('MT-') 
    sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)
    adata = adata[adata.obs.n_genes_by_counts < 10000, :]
    adata = adata[adata.obs.pct_counts_mt < 20, :]
    adata = adata[adata.obs.total_counts < 50000, :]
    sc.pp.normalize_total(adata, target_sum=1e5)
    sc.pp.log1p(adata)

    print(f"Processed data shape: {adata.X.shape}")
    return adata

def setup_directories(args):
    """Set up necessary directories."""
    if not os.path.exists(args['model_dir']):
        os.makedirs(args['model_dir'])
        print(f"Directory created: {args['model_dir']}")
    else:
        print(f"Directory already exists: {args['model_dir']}")

    args['model_data_dir'] = os.path.join(args['model_dir'], 'processed_data')
    if not os.path.exists(args['model_data_dir']):
        os.mkdir(args['model_data_dir'])
        print(f"Directory created: {args['model_data_dir']}")
    else:
        print(f"Directory already exists: {args['model_data_dir']}")

def save_processed_data(args, all_data_df):
    """Save processed data to a pickle file."""
    model_data_file = os.path.join(args['model_data_dir'], f"{args['model_prefix']}.pkl")
    with open(model_data_file, 'wb') as file:
        pickle.dump(all_data_df, file)
    print(f"Processed data saved to {model_data_file}")

def main(args):
    # Load and preprocess data
    adata = load_data(args)
    
    # Preprocess data
    data_x_s, mask_list_dict, args, seq_gene_df, seq_gene_df_su, TP_df, ppi_df_1, data = pre_process(args, adata)
    
    # Set up directories
    setup_directories(args)
    
    # Prepare data for saving
    all_data_df = {
        'data_x_s': data_x_s,
        'mask_dict': mask_list_dict,
        'args': args,
        'seq_gene_df': seq_gene_df,
        'seq_gene_df_su': seq_gene_df_su,
        'TP_df': TP_df,
        'ppi_df_1': ppi_df_1,
        'data': data
    }
    
    # Save processed data
    save_processed_data(args, all_data_df)
    
    # Perform k-fold cross-validation
    kfold = KFoldTorch(args)
    kfold.train_kfold(iGTP, mask_list_dict, data_x_s)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run IGTP KFold PBMC analysis")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML configuration file")
    
    args = parser.parse_args()
    
    # Load configuration from YAML file
    config = load_config(args.config)
    
    # Set model prefix
    config['model_prefix'] = f"{os.path.basename(config['tp_file_path']).replace('.gmt', '')}_{os.path.basename(config['ppi_file_path']).replace('.txt', '')}_{config['recon_loss']}_{config['init_type']}_{config['vb_nu']}batch"
    
    main(config)
