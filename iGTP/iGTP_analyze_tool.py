import numpy as np
import pandas as pd
from tqdm import tqdm
import multiprocessing
from functools import partial
import torch
import os
from model.iGTP_Linear import *
from iGTP_model import *
from preprocess import *
import pickle
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from adjustText import adjust_text

def load_and_evaluate_model(ex_args):
    """
    Load the iGTP model, evaluate it, and generate encodings.
    """
    model_data_file = ex_args['processed_data_path']
    with open(model_data_file, 'rb') as file:
        task_file = pickle.load(file)
    
    args=task_file['args']
    args.update(ex_args)
    args['best_model_path']=args['model_dir']+args['model_prefix']+'best_fold.pt'
    data_x_s=task_file['data_x_s']
    mask_list_dict=task_file['mask_dict']
    data=task_file['data']
    TP_df=task_file['TP_df']
    PPI_df=task_file['ppi_df_1']

    output_folder=args['output_folder']
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    model = iGTP(args, mask_list_dict)
    model_file_path=args['best_model_path']
    saved_state_dict = torch.load( model_file_path, map_location="cpu")
    model.load_state_dict(saved_state_dict)
    model.eval()
    
    with torch.no_grad():
        TP_emb = model.encode_z(torch.tensor(data_x_s).float())
        cell_gene=model.encode_ppi(torch.tensor(data_x_s).float())
    TP_path=os.path.join(output_folder,'TP_emb.npy')
    np.save(TP_path,TP_emb)
    cell_gene_path=os.path.join(output_folder,'cell_gene.npy')
    np.save(cell_gene_path,cell_gene)
    gene_weight=saved_state_dict['decoder_Gene_Gene.0.weight'].cpu().detach().numpy()
    tsne = TSNE(n_components=2, random_state=42).fit_transform(TP_emb)
    cell_label_df = data.obs.reset_index()
    cell_label_df['dim_1'] = tsne[:, 0]
    cell_label_df['dim_2'] = tsne[:, 1]  
    cell_label_df['cell_index'] = range(len(cell_label_df))
    cell_label_df_path=os.path.join(output_folder,'cell_label_df.csv')
    print(cell_label_df_path)
    cell_label_df.to_csv(cell_label_df_path,index=False)
    TP_df['TP_index'] = range(len(TP_df))
    TP_df_save_path=os.path.join(output_folder,'TP_df.csv')
    print(TP_df_save_path)
    TP_df.to_csv(TP_df_save_path,index=False)
    return model,args, TP_emb, cell_gene ,gene_weight, cell_label_df,TP_df,PPI_df

def compute_and_save_bayes_factor(model, args,TP_emb, cell_label_df, TP_df, codiction_column, cell_type_column, 
                                compare_condiciton, tsne, prefix,m_permutation=12000):
    """
    Compute Bayes factor if conditions are met, and save the results.

    Parameters:
    - model: The model object with compute_pair_bayes_factor method
    - TP_emb: TP embeddings
    - label_df: DataFrame containing labels
    - TP_df: DataFrame containing TP information
    - codiction_column: Name of the condition column
    - cell_type_column: Name of the cell type column
    - compare_condiciton: List of two conditions to compare
    - tsne: TSNE object
    - output_folder: Path to save the output
    - m_permutation: Number of permutations (default 12000)

    Returns:
    - Path to the saved file if computation was performed, None otherwise
    """
    if codiction_column in cell_label_df.columns and len(compare_condiciton) == 2:
        res = model.compute_pair_bayes_factor(
            TP_emb, 
            cell_label_df, 
            codiction_column, 
            cell_type_column, 
            compare_condiciton[0], 
            compare_condiciton[1], 
            tsne, 
            m_permutation=m_permutation
        )
        os.makedirs(os.path.join(args['output_folder'],prefix), exist_ok=True)
        TP_df_b = pd.DataFrame(res)
        TP_df_b['TP_name'] = TP_df['TP_name'].values
        TP_df_b['TP_gene'] = TP_df['TP_gene'].values
        output_folder = args['output_folder']
        file_name = compare_condiciton[0]+'_'+compare_condiciton[1]+'.csv'
        TP_df_save_path = os.path.join(output_folder,prefix, file_name)
        TP_df_b.to_csv(TP_df_save_path, index=False)
        file_name = compare_condiciton[0]+'_'+compare_condiciton[1]+'.pdf'
        TP_df_plot_save_path = os.path.join(output_folder,prefix, file_name)
        bayes_factor_m = TP_df_b['bayes_factor'].mean()
        bayes_factor_s = TP_df_b['bayes_factor'].std()
        bayes_factor_m_t=bayes_factor_m+2*bayes_factor_s
        mad_m = TP_df_b['mad'].mean()
        mad_s = TP_df_b['mad'].std()
        mad_m_t = mad_m+2*mad_s

        plot_dfe(res,
            sig_lvl=bayes_factor_m_t,
            pathway_list=list(TP_df['TP_name'].values),
            lfc_lvl=mad_m_t,
            to_plot=None,
            figsize=[14,7],
            s=30,
            fontsize=20,
            textsize=12,
            title= compare_condiciton[0]+'V_S'+compare_condiciton[1]+'_volcano plot',
            save=TP_df_plot_save_path)
        return TP_df_b, [bayes_factor_m_t, mad_m_t]
    else:
        print("The 'condition' column does not exist in label_df or the length of 'compare_condiciton' is not 2.")
        return None,None

def plot_tsne(cell_label_df, args, select_column):
    """
    Create a tSNE plot based on a selected column and save it as a PDF file.

    Parameters:
    -----------
    cell_label_df : pandas.DataFrame
        DataFrame containing the tSNE dimensions and cell label data.
        Must include columns 'dim_1', 'dim_2', and the column specified by select_column.
    args : dict
        Dictionary containing configuration parameters.
        Must include 'output_folder' key.
    select_column : str
        Name of the column in cell_label_df to use for coloring the scatter plot.

    Returns:
    --------
    str or None
        Path to the saved figure if successful, None otherwise.

    Raises:
    -------
    KeyError
        If 'output_folder' is not in args or select_column is not in cell_label_df.
    """
    # Set up the save folder
    save_folder = os.path.join(args['output_folder'], 'tSNE')
    os.makedirs(save_folder, exist_ok=True)

    if select_column in cell_label_df.columns:
        # Set up the plot title and style
        title = f'{select_column}_tSNE'
        sns.set_palette('colorblind')
        
        # Create the plot
        fig, ax = plt.subplots(figsize=(7, 7))
        sns.scatterplot(x='dim_1', y='dim_2', hue=select_column, data=cell_label_df, 
                        linewidth=0, alpha=1, s=3, ax=ax)

        # Customize the legend
        ax.legend(title=title, title_fontsize=20, loc='upper center',
                  bbox_to_anchor=(0.5, -0.1), ncol=4, borderaxespad=0.,
                  fontsize=12, frameon=False, markerscale=5)

        # Set labels and ticks
        ax.set_xlabel('tSNE-1', fontsize=15)
        ax.set_ylabel('tSNE-2', fontsize=15)
        ax.tick_params(axis='both', which='major', labelsize=15)

        # Save the figure
        file_name = os.path.join(save_folder, f'{title}.pdf')
        plt.savefig(file_name, format='pdf', dpi=300, bbox_inches='tight')
        plt.close(fig)

        return file_name
    else:
        print(f"Error: '{select_column}' not found in the provided DataFrame.")
        return None

def plot_tsne_emb_color(cell_label_df, args, TP_df, TP_emb):
    """
    Create tSNE plots colored by pathway embeddings and save them as PDF files.

    Parameters:
    -----------
    cell_label_df : pandas.DataFrame
        DataFrame containing the tSNE dimensions.
        Must include columns 'dim_1' and 'dim_2'.
    args : dict
        Dictionary containing configuration parameters.
        Must include 'output_folder' key.
    TP_df : pandas.DataFrame
        DataFrame containing pathway information.
        Must include columns 'TP_name' and 'TP_index'.
    TP_emb : numpy.ndarray
        2D array of pathway embeddings, where each column corresponds to a pathway.

    Returns:
    --------
    list
        List of paths to the saved figures.

    Raises:
    -------
    KeyError
        If required columns are not in the DataFrames or 'output_folder' is not in args.
    ValueError
        If the shapes of TP_df and TP_emb don't match.
    """
    # Set up the save folder
    save_folder = os.path.join(args['output_folder'], 'tSNE_emb')
    os.makedirs(save_folder, exist_ok=True)

    saved_paths = []

    # Validate input shapes
    if len(TP_df) != TP_emb.shape[1]:
        raise ValueError("Number of pathways in TP_df doesn't match TP_emb columns.")

    for j in range(len(TP_df)):
        pathway_name = TP_df.TP_name.values[j]
        pathway_nu = TP_df.TP_index.values[j]

        # Create the plot
        fig, ax = plt.subplots(figsize=(7, 7))
        scatter = ax.scatter(cell_label_df['dim_1'].values, cell_label_df['dim_2'].values,
                             alpha=0.8, linewidths=0, c=TP_emb[:, pathway_nu],
                             marker='o', s=3, cmap='seismic')

        # Set title and labels
        ax.set_title(pathway_name, fontsize=15)
        ax.set_xlabel('tSNE-1', fontsize=15)
        ax.set_ylabel('tSNE-2', fontsize=15)

        # Adjust tick font sizes
        ax.tick_params(axis='both', which='major', labelsize=15)

        # Add and customize colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.ax.tick_params(labelsize=15)

        # Save the figure
        save_path = os.path.join(save_folder, f"{pathway_name}_colored_tSNE.pdf")
        plt.savefig(save_path, format='pdf', dpi=300, bbox_inches='tight')
        plt.close(fig)

        saved_paths.append(save_path)

    return saved_paths

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

def get_PPI_from_TP_threshold(TP_df_b, PPI_df, bayes_factor_threshold=4, mad_threshold=3):
    """
    Process Transcription Program (TP) and Protein-Protein Interaction (PPI).
    
    Args:
    TP_df_b (pd.DataFrame): DataFrame containing TP data with Bayes factors.
    PPI_df (pd.DataFrame): DataFrame containing PPI data.
    bayes_factor_threshold (float): Threshold for absolute Bayes factor (default: 4).
    mad_threshold (float): Threshold for MAD (Median Absolute Deviation) (default: 3).
    
    Returns:
    tuple: (TP_df_b_1, new_PPI_df, se_g)
        TP_df_b_1 (pd.DataFrame): Filtered TP DataFrame.
        new_PPI_df (pd.DataFrame): Filtered and processed PPI DataFrame.
        selected_genes (list): List of unique genes from filtered TPs.
    """
    
    # Process TP data
    TP_df_b['abs_b'] = abs(TP_df_b.bayes_factor)
    TP_df_b = TP_df_b.sort_values(by='abs_b', ascending=False)
    TP_df_b_1 = TP_df_b[(TP_df_b.abs_b >= bayes_factor_threshold) & (TP_df_b['mad'] >= mad_threshold)]
    
    # Extract unique genes from filtered TPs
    selected_genes = list(set([gene for tp_genes in TP_df_b_1['TP_gene'] for gene in tp_genes]))
    
    # Process PPI data
    new_PPI_df = PPI_df[
        (PPI_df.gene1.isin(selected_genes)) & 
        (PPI_df.gene2.isin(selected_genes)) & 
        (PPI_df.gene1 != PPI_df.gene2)
    ].copy()
    
    new_PPI_df['PPI'] = new_PPI_df['gene1'] + '-' + new_PPI_df['gene2']
    new_PPI_df['PPI_index'] = range(len(new_PPI_df))
    
    return new_PPI_df

def select_tp_and_get_ppi_indices(TP_df, PPI_df, tp_names):
    """
    Select interested Transcription Programs (TPs) and get relevant PPI indices.
    
    Args:
    TP_df (pd.DataFrame): DataFrame containing TP data.
    PPI_df (pd.DataFrame): DataFrame containing PPI data.
    tp_names (list): List of TP names of interest.
    
    Returns:
    tuple: (selected_TP_df, relevant_PPI_indices)
        selected_TP_df (pd.DataFrame): DataFrame of selected TPs.
        relevant_PPI_indices (list): List of relevant PPI indices.
    """
    
    # Select interested TPs
    selected_TP_df = TP_df[TP_df['TP_name'].isin(tp_names)]
    
    # Get unique genes from selected TPs
    selected_genes = set([gene for tp_genes in selected_TP_df['TP_gene'] for gene in tp_genes])
    
    # Get relevant PPI indices
    new_PPI_df = PPI_df[
        (PPI_df.gene1.isin(selected_genes)) & 
        (PPI_df.gene2.isin(selected_genes)) & 
        (PPI_df.gene1 != PPI_df.gene2)
    ].copy()
    
    return new_PPI_df

def process_chunk(chunk, gene_weight, cell_label_df,cell_index, cell_gene, output_dir, device='cpu'):
    """
    Process a chunk of PPI data, calculating interaction scores and saving results.
    
    Args:
    chunk (pd.DataFrame): A subset of PPI data to process.
    gene_weight (np.array): Weight matrix for genes.
    cell_label_df (pd.DataFrame): DataFrame containing cell labels.
    cell_index (np.array): Cell index for each cell in the cell_label_df.
    cell_gene (np.array): Cell-gene expression matrix.
    output_dir (str): Directory to save output files.
    device (str): 'cpu' or 'cuda' for processing.
    """
    # Move data to GPU if using CUDA
    if device == 'cuda':
        gene_weight = torch.tensor(gene_weight, device=device)
        cell_gene = torch.tensor(cell_gene, device=device)
    
    for _, row in tqdm(chunk.iterrows(), total=len(chunk)):
        gene_1_index, gene_2_index = row['gene_1_index'], row['gene_2_index']
        PPI = row['PPI']
        
        if device == 'cuda':
            # Create masks for the specific gene pair
            mask1 = torch.zeros_like(gene_weight, dtype=torch.bool, device=device)
            mask2 = torch.zeros_like(gene_weight, dtype=torch.bool, device=device)
            mask1[gene_1_index, gene_2_index] = True
            mask2[gene_2_index, gene_1_index] = True
            
            # Apply masks to gene weights
            gene_weight1 = torch.where(mask1, gene_weight, torch.tensor(0., device=device))
            gene_weight2 = torch.where(mask2, gene_weight, torch.tensor(0., device=device))
            
            # Calculate interaction scores
            value1 = torch.sum(torch.matmul(cell_gene, gene_weight1.T), dim=1)
            value2 = torch.sum(torch.matmul(cell_gene, gene_weight2.T), dim=1)
            
            temp_df = pd.DataFrame({
                'cell': cell_label_df[cell_index].values,
                'value1': value1.cpu().numpy(),
                'value2': value2.cpu().numpy()
            })
        else:
            # CPU version of the same operations
            mask1 = np.zeros_like(gene_weight, dtype=bool)
            mask2 = np.zeros_like(gene_weight, dtype=bool)
            mask1[gene_1_index, gene_2_index] = True
            mask2[gene_2_index, gene_1_index] = True
            
            gene_weight1 = np.where(mask1, gene_weight, 0)
            gene_weight2 = np.where(mask2, gene_weight, 0)
            
            value1 = np.sum(np.matmul(cell_gene, gene_weight1.T), axis=1)
            value2 = np.sum(np.matmul(cell_gene, gene_weight2.T), axis=1)
            
            temp_df = pd.DataFrame({
                'cell': cell_label_df[cell_index].values,
                'value1': value1,
                'value2': value2
            })
        
        # Take the maximum of value1 and value2 as the final score
        temp_df[PPI] = np.where(temp_df['value1'] > temp_df['value2'], temp_df['value1'], temp_df['value2'])
        temp_df = temp_df[['cell', PPI]]
        
        # Save results to CSV
        output_path = os.path.join(output_dir, f"{PPI}.csv")
        temp_df.to_csv(output_path, index=False)

def ppi_parallel_process(PPI_df, gene_weight, cell_label_df, cell_index ,cell_gene, num_processes=1, use_gpu=False, output_dir='./PPI/PBMC/'):
    """
    Process PPI data in parallel, optionally using GPU acceleration.
    
    Args:
    df (pd.DataFrame): PPI data to process.
    gene_weight (np.array): Weight matrix for genes.
    label_df (pd.DataFrame): DataFrame containing cell labels.
    cell_gene (np.array): Cell-gene expression matrix.
    num_processes (int): Number of CPU processes to use if not using GPU.
    use_gpu (bool): Whether to use GPU acceleration.
    output_dir (str): Directory to save output files.
    """
    device = 'cuda' if use_gpu and torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    if device == 'cuda':
        # Process entire dataset at once on GPU
        process_chunk(PPI_df, gene_weight, cell_label_df,cell_index, cell_gene, output_dir, device)
    else:
        # Split data and use multiprocessing on CPU
        chunks = np.array_split(PPI_df, num_processes)
        with multiprocessing.Pool(processes=num_processes) as pool:
            pool.map(partial(process_chunk,
                             gene_weight=gene_weight,
                             cell_index=cell_index,
                             label_df=cell_label_df,
                             cell_gene=cell_gene,
                             output_dir=output_dir,
                             device=device), chunks)

def collect_ppi_results(PPI_df, output_dir='./PPI/PBMC/'):
    """
    Collect and combine PPI processing results.
    
    Args:
    new_PPI_df (pd.DataFrame): DataFrame containing information about processed PPIs.
    output_dir (str): Directory where PPI results are saved.
    
    Returns:
    np.array: Combined PPI scores for all cells and interactions.
    """
    z_ppi = []
    for PPI in tqdm(PPI_df.PPI.values):
        temp_df = pd.read_csv(os.path.join(output_dir, f"{PPI}.csv"))
        z_ppi.append(temp_df[PPI].values)
    return np.array(z_ppi).T

# +

def create_cell_type_condition_heatmap(z, cell_label_df, cell_type_column, index_df, 
                                    index_column, index_name_column, condition_column,
                                    conditions, output_path='./fig_3_result/cluster_heatmap/brain_ad.pdf'):
    """
    Create a clustered heatmap comparing cell types across different conditions.

    Parameters:
    - z: numpy array, the embedding matrix
    - cell_label_df: pandas DataFrame containing cell information
    - cell_type_column: str, name of the column in cell_label_df that contains cell type information
    - index_df: pandas DataFrame containing index information (e.g., TP information)
    - index_column: str, name of the column in index_df that contains the index values
    - index_name_column: str, name of the column in index_df that contains the index names
    - condition_column: str, name of the column in cell_label_df that contains condition information
    - conditions: list of str, conditions to compare
    - output_path: str, path to save the output heatmap
    """

    # Initialize list to store DataFrames for each condition
    all_emb_dfs = []

    # Iterate through each condition
    for condition in conditions:
        all_emb = []
        # Iterate through each unique cell type
        for cell_type in cell_label_df[cell_type_column].unique():
            # Create mask for cells of this type and condition
            mask = (cell_label_df[cell_type_column] == cell_type) & (cell_label_df[condition_column] == condition)
            
            # Extract relevant embeddings
            emb_file1 = z[:, index_df[index_column].values]
            emb_file1 = emb_file1[mask,]
            
            # Calculate mean embedding or use zeros if no cells found
            if emb_file1.shape[0] > 0:
                emb_file1 = np.mean(emb_file1, axis=0)
            else:
                emb_file1 = np.zeros(emb_file1.shape[1])
            
            all_emb.append(emb_file1)
        
        # Convert to DataFrame
        all_emb = np.array(all_emb)
        all_emb_df = pd.DataFrame(all_emb, index=cell_label_df[cell_type_column].unique())
        all_emb_df.columns = index_df[index_name_column].values
        
        all_emb_dfs.append(all_emb_df)
    
    # Concatenate DataFrames for all conditions
    all_emb_df = pd.concat(all_emb_dfs, keys=conditions, names=['Condition', 'Cell Type'])
    all_emb_df = all_emb_df.fillna(0).replace([np.inf, -np.inf], 0)

    # Create a color palette for conditions
    condition_colors = list(sns.color_palette("husl", n_colors=len(conditions)).as_hex())
    condition_lut = dict(zip(conditions, condition_colors))
   
    # Create row colors based on conditions
    row_colors = all_emb_df.index.get_level_values('Condition').map(condition_lut)
    
    # Create the clustermap
    plt.figure(figsize=(20, 20))
    myplot = sns.clustermap(all_emb_df, 
                            row_cluster=False,  # Cluster rows
                            col_cluster=False,  # Cluster columns (TP indices)
                            method='complete',  # Linkage method for hierarchical clustering
                            metric='correlation',  # Distance metric for clustering
                            cmap="seismic",  # Color scheme 
                            z_score=0,  # Standardize the data
                            figsize=(20, 20),  # Figure size
                            row_colors=row_colors,  # Add row colors for conditions
                            colors_ratio=0.02,  # Adjust the width of the color bar
                            dendrogram_ratio=(0.1, 0.2))  # Adjust the size of dendrograms
    
    # Adjust x-axis labels
    myplot.ax_heatmap.set_xticklabels(myplot.ax_heatmap.get_xticklabels(),
                                      rotation=45, ha='right')  # Rotate labels for readability
    
    # Clean up y-axis labels
    y_labels = myplot.ax_heatmap.get_yticklabels()
    cleaned_labels = [label.get_text().replace('stimulated-', '').replace('un_', '') for label in y_labels]
    print(cleaned_labels)

    # Set cleaned y-axis labels
    myplot.ax_heatmap.set_yticklabels(cleaned_labels,
                                      rotation=0, ha='left', va='center')
    
    # Move y-axis labels to the right side
    myplot.ax_heatmap.yaxis.set_label_position("right")
    myplot.ax_heatmap.yaxis.tick_right()

    # Set font sizes
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    
    # Add a legend for conditions
    handles = [plt.Rectangle((0,0),1,1, color=condition_lut[label]) for label in conditions]
    plt.legend(handles, conditions, title="Conditions", 
               bbox_to_anchor=(1, 1), bbox_transform=plt.gcf().transFigure, loc="upper right")
    
    # Save the figure
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()  # Close the figure to free up memory
    
    print(f"Heatmap saved to {output_path}")


# -

def create_pairwise_scatter_plots(z, cell_label_df, cell_type_column, index_df, 
                                  index_column, index_name_column, color_dict, 
                                  condition_column, output_dir='./fig_3_result/',
                                  output_df_dir='TP_shift_brain_df/',
                                  output_sc_dir='TP_shift_brain/',
                                  sample_frac=0.5):
    """
    Create pairwise scatter plots for transcription programs (TPs) across different cell types and conditions.
    
    Args:
    z (np.array): The embedding matrix, where rows are cells and columns are TPs.
    cell_label_df (pd.DataFrame): DataFrame containing cell information including cell type and condition.
    cell_type_column (str): Name of the column in cell_label_df that contains cell type information.
    index_df (pd.DataFrame): DataFrame containing TP information.
    index_column (str): Name of the column in index_df that contains the TP indices.
    index_name_column (str): Name of the column in index_df that contains the TP names.
    color_dict (dict): Dictionary mapping conditions to colors for the plot.
    condition_column (str): Name of the column in cell_label_df that contains condition information.
    output_dir (str): Base directory to save output files (default: './fig_3_result/').
    output_df_dir (str): Subdirectory for saving DataFrames (default: 'TP_shift_brain_df/').
    output_sc_dir (str): Subdirectory for saving scatter plots (default: 'TP_shift_brain/').
    sample_frac (float): Fraction of data to sample for plotting (default: 0.5).
    
    Returns:
    None: The function saves data files and scatter plots to the specified output directories.
    """
    
    # Ensure output directories exist
    os.makedirs(os.path.join(output_dir, output_df_dir), exist_ok=True)
    os.makedirs(os.path.join(output_dir, output_sc_dir), exist_ok=True)
    
    # Extract TP names and indices
    int_path_name_1 = index_df[index_name_column].values
    int_path_index = index_df[index_column].values
    
    # Iterate over all pairs of TPs
    for m in tqdm(range(len(int_path_name_1))):
        for n in range(m+1, len(int_path_name_1)):
            name_pair = [int_path_name_1[m], int_path_name_1[n]]
            index_pair = [int_path_index[m], int_path_index[n]]
            
            # Create DataFrame for this pair of TPs
            temp_df = cell_label_df[[cell_type_column, condition_column]].copy()
            temp_df[name_pair[0]] = z[:, index_pair[0]]  # Add values for first TP
            temp_df[name_pair[1]] = z[:, index_pair[1]]  # Add values for second TP
            
            # Save DataFrame
            df_path = os.path.join(output_dir, output_df_dir, f"{name_pair[0]}_{name_pair[1]}.csv")
            temp_df.to_csv(df_path, index=False)
            
            # Create scatter plot
            plt.figure(figsize=(50, 10))
            g = sns.FacetGrid(temp_df.sample(frac=sample_frac),  # Sample data for faster plotting
                              col=cell_type_column,  # Separate plots for each cell type
                              hue=condition_column,  # Color points by condition
                              margin_titles=True,
                              palette=color_dict)  # Use provided color dictionary
            
            # Add scatter plots to the FacetGrid
            g.map_dataframe(sns.scatterplot, x=name_pair[0], y=name_pair[1], s=0.8)
            
            # Save scatter plot
            plot_path = os.path.join(output_dir, output_sc_dir, f"{name_pair[0]}_{name_pair[1]}_shift.png")
            g.figure.savefig(plot_path, dpi=200)
            plt.close(g.figure)  # Close the figure to free up memory
    
    print(f"Pairwise scatter plots and data files saved in {output_dir}")
