from collections import OrderedDict
import torch
import numpy as np
import scanpy as sc
from scipy import sparse
import os
from tqdm import tqdm
import pandas as pd

def read_gmt(fname, seq_gene, tp_fraction=None, sep='\t', min_g=0, max_g=5000):
    """
    Read GMT file into dictionary of pathway:genes.
    
    Args:
        fname (str): Path to the GMT file.
        seq_gene (list): List of genes in the sequence data.
        tp_fraction (float): Minimum fraction of overlap required.
        sep (str): Separator used in the GMT file.
        min_g (int): Minimum number of genes in a pathway.
        max_g (int): Maximum number of genes in a pathway.
    
    Returns:
        dict_pathway (OrderedDict): Dictionary of pathways and their genes.
        all_gene (list): List of all genes found in the pathways.
    """
    dict_pathway = OrderedDict()
    if tp_fraction is None:
        tp_fraction = 0.0
    all_gene = []  
    
    with open(fname) as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            val = line.split(sep)
            if min_g <= len(val[2:]) <= max_g:
                over_lape_gene_frac = len(set(seq_gene).intersection(set(val[2:]))) / len(val[2:])
                if over_lape_gene_frac >= tp_fraction:
                    dict_pathway[val[0]] = list(set(seq_gene).intersection(set(val[2:])))
                    all_gene.extend(list(set(seq_gene).intersection(set(val[2:]))))
    
    all_gene = list(set(all_gene))
    return dict_pathway, all_gene

def pre_process(args, data):
    """
    Preprocess the data for model training.
    
    Args:
        args (dict): Dictionary of arguments.
        data (AnnData): Annotated data matrix.
    
    Returns:
        tuple: Processed data and related information.
    """
    tp_file_path = args['tp_file_path']
    ppi_file_path = args['ppi_file_path']
    tp_overlap_fraction = args['tp_overlap_fraction']
    assert 0 <= tp_overlap_fraction <= 1, "Error: 'tp_overlap_fraction' must be between 0 and 1 (inclusive)."

    # Get data seq gene
    seq_gene = list(data.var.index.values)

    # Get TP file and filter by the overlap number of seq gene
    gmt_file, gmt_co_gene = read_gmt(tp_file_path, seq_gene, tp_fraction=tp_overlap_fraction)
    print(f"Number of genes in TP: {len(gmt_co_gene)}")

    # Process gene information
    seq_gene_df = pd.DataFrame(seq_gene, columns=['gene'])
    seq_gene_df['gene_index'] = range(len(seq_gene_df))
    seq_gene_df['in_tp'] = np.where(seq_gene_df['gene'].isin(gmt_co_gene), 1, 0)
    seq_gene_df = seq_gene_df[seq_gene_df.in_tp == 1]
    
    # Process data matrix
    try:
        data_x_s = data.X.A
    except:
        data_x_s = data.X
    
    data_x_s = data_x_s[:, seq_gene_df['gene_index']]
    
    seq_gene_df['gene_index'] = range(len(seq_gene_df))
    seq_gene = list(seq_gene_df['gene'].values)
    seq_gene_dict = {gene: index for index, gene in enumerate(seq_gene_df['gene'].values)}

    # Process GMT file
    new_gmt_file = {TP: [seq_gene_dict[j] for j in TP_gene] for TP, TP_gene in gmt_file.items()}

    TP_df = pd.DataFrame(list(gmt_file.keys()), columns=['TP_name'])
    TP_df['TP_gene'] = list(gmt_file.values())
    TP_df['TP_index'] = range(len(TP_df))

    # Process PPI file
    ppi_df = pd.read_csv(ppi_file_path, delimiter='\t')
    ppi_df = ppi_df[ppi_df.gene1.isin(seq_gene) & ppi_df.gene2.isin(seq_gene)]
    print(f'Number of Gene {len(set(list(ppi_df.gene1.unique()) + list(ppi_df.gene2.unique())))} in PPI')
    
    ppi_df = pd.concat([ppi_df, ppi_df.rename(columns={'gene1': 'gene2', 'gene2': 'gene1'}),
                        pd.DataFrame({'gene1': seq_gene, 'gene2': seq_gene})])
    ppi_df['gene_1_index'] = ppi_df['gene1'].map(seq_gene_dict)
    ppi_df['gene_2_index'] = ppi_df['gene2'].map(seq_gene_dict)
    ppi_df = ppi_df.sort_values(by='gene_1_index')

    # Create PPI dict
    new_ppi_dict = {i: ppi_df[ppi_df.gene_1_index == i].gene_2_index.values for i in range(len(seq_gene))}

    # Create TP-Gene mask
    print('Create TP->Gene mask')
    TP_gene_mask = np.array([[1 if j in TP_Gene else 0 for j in range(len(seq_gene))] for TP_Gene in new_gmt_file.values()])
    print(f'Number of TP {TP_gene_mask.shape[0]}, and Gene {TP_gene_mask.shape[1]}')

    # Create Gene-Gene mask
    print('Create Gene->Gene mask')
    gene_gene_mask = np.array([[1 if j in TP_Gene else 0 for j in range(len(seq_gene))] for TP_Gene in new_ppi_dict.values()])
    print(f'Number of Gene {gene_gene_mask.shape[0]}, and Gene {gene_gene_mask.shape[1]}')

    mask_list_dict = {
        'TP_Gene_mask': TP_gene_mask,
        'Gene_Gene_mask': gene_gene_mask
    }

    args['encoder_layer_list'] = [len(gmt_co_gene)] + args['encoder_layer_list']
    args['n_TP'] = len(list(gmt_file.keys()))

    seq_gene_df_su = seq_gene_df.groupby('in_tp')['gene'].count().reset_index()
    seq_gene_df_su['in_tp'] = np.where(seq_gene_df_su['in_tp'] == 1, 'Yes', 'No')
    for _, row in seq_gene_df_su.iterrows():
        print(f"The number of gene in TP {row['in_tp']} was {row['gene']:.2f}")

    if sparse.issparse(data_x_s):
        print(f"Converting sparse matrix of shape {data_x_s.shape} to dense array...")
        data_x_s = data_x_s.toarray()
        print(f"Conversion complete. Memory usage: {data_x_s.nbytes / (1024**2):.2f} MB")

    return data_x_s, mask_list_dict, args, seq_gene_df, seq_gene_df_su, TP_df, ppi_df, data

# The pre_process_test function is omitted for brevity, but it would be annotated similarly