import os
import yaml
from igtp.analyze_tool import (
    load_and_evaluate_model,
    plot_tsne,
    plot_tsne_emb_color,
    compute_and_save_bayes_factor,
    create_cell_type_condition_heatmap
)

def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def evaluate_model(args):
    """
    Evaluate a trained iGTP model using the provided configuration.
    
    Args:
        args (dict): Configuration dictionary or path to YAML config file
            Must contain:
            - 'processed_data_path': Path to the processed data pickle file
            - 'output_folder': Directory to save results
            - 'cell_type_column': Column name for cell types in the dataset
            - 'condition_column': Column name for conditions/treatments in the dataset
            
    Returns:
        dict: Evaluation results including model, embeddings, and data frames
    """
    # Handle config if it's a file path
    if isinstance(args, str):
        args = load_config(args)
    
    # Ensure output directory exists
    os.makedirs(args['output_folder'], exist_ok=True)
    print(f"Output will be saved to {args['output_folder']}")
    
    # Load the model and generate embeddings
    print(f"Loading model from {args['processed_data_path']}")
    model, model_args, TP_emb, cell_gene, gene_weight, cell_label_df, TP_df, PPI_df = load_and_evaluate_model(args)
    
    # Get required column names from config
    cell_type_column = args.get('cell_type_column')
    condition_column = args.get('condition_column')
    
    # Verify that required columns exist in the data
    if cell_type_column and cell_type_column not in cell_label_df.columns:
        print(f"Warning: Specified cell_type_column '{cell_type_column}' not found in data.")
        print(f"Available columns: {list(cell_label_df.columns)}")
        cell_type_column = None
    
    if condition_column and condition_column not in cell_label_df.columns:
        print(f"Warning: Specified condition_column '{condition_column}' not found in data.")
        print(f"Available columns: {list(cell_label_df.columns)}")
        condition_column = None
    
    # Create t-SNE plots for specified columns
    print("Creating t-SNE plots")
    tsne_paths = {}
    
    # Plot t-SNE for cell type column if available
    if cell_type_column:
        print(f"Creating t-SNE plot for cell type column: {cell_type_column}")
        file_path = plot_tsne(cell_label_df, model_args, cell_type_column)
        tsne_paths[cell_type_column] = file_path
    
    # Plot t-SNE for condition column if available
    if condition_column:
        print(f"Creating t-SNE plot for condition column: {condition_column}")
        file_path = plot_tsne(cell_label_df, model_args, condition_column)
        tsne_paths[condition_column] = file_path
    
    # Plot t-SNE colored by TP embeddings
    print("Creating t-SNE plots colored by TP embeddings")
    tp_tsne_paths = plot_tsne_emb_color(cell_label_df, model_args, TP_df, TP_emb)
    
    # Store results from analyses
    bayes_factor_results = {}
    heatmap_paths = {}
    
    # Perform differential expression analysis if both required columns are available
    if cell_type_column and condition_column:
        print(f"Using cell type column: {cell_type_column} and condition column: {condition_column}")
        
        # Get unique conditions
        condition_list = list(cell_label_df[condition_column].unique())
        print(f"Found {len(condition_list)} unique conditions: {condition_list}")
        
        # Simple condition-based analysis
        print("Performing condition-based analysis")
        
        # Get the global t-SNE coordinates
        tsne = cell_label_df[['dim_1', 'dim_2']].values
        
        # If there are two conditions, compare them directly
        if len(condition_list) == 2:
            print(f"Computing Bayes factor for {condition_list[0]} vs {condition_list[1]}")
            bf_result, thresholds = compute_and_save_bayes_factor(
                model, model_args, TP_emb, cell_label_df, TP_df, 
                condition_column, cell_type_column, condition_list, 
                tsne, prefix="condition_comparison", m_permutation=12000
            )
            key = f"{condition_list[0]}_vs_{condition_list[1]}"
            bayes_factor_results[key] = {
                'result': bf_result,
                'thresholds': thresholds
            }
        # Otherwise, do pairwise comparisons
        else:
            for m in range(len(condition_list)):
                for n in range(m+1, len(condition_list)):
                    compare_pair = [condition_list[m], condition_list[n]]
                    print(f"Computing Bayes factor for {compare_pair[0]} vs {compare_pair[1]}")
                    bf_result, thresholds = compute_and_save_bayes_factor(
                        model, model_args, TP_emb, cell_label_df, TP_df, 
                        condition_column, cell_type_column, compare_pair, 
                        tsne, f"{compare_pair[0]}_vs_{compare_pair[1]}", m_permutation=12000
                    )
                    key = f"{compare_pair[0]}_vs_{compare_pair[1]}"
                    bayes_factor_results[key] = {
                        'result': bf_result,
                        'thresholds': thresholds
                    }
            
        # Create overall heatmap
        output_path = os.path.join(model_args['output_folder'], 'cluster_heatmap.pdf')
        print(f"Creating cluster heatmap")
        create_cell_type_condition_heatmap(
            TP_emb, cell_label_df, cell_type_column, TP_df, 
            'TP_index', 'TP_name', condition_column, condition_list, 
            output_path=output_path
        )
        heatmap_paths['overall'] = output_path
    else:
        print("Could not find both condition and cell type columns for differential analysis.")
        print(f"Available columns: {list(cell_label_df.columns)}")
        print("Please specify 'cell_type_column' and 'condition_column' in the config.")
    
    # Return evaluation results
    results = {
        'model': model,
        'args': model_args,
        'TP_emb': TP_emb,
        'cell_gene': cell_gene,
        'gene_weight': gene_weight,
        'cell_label_df': cell_label_df,
        'TP_df': TP_df,
        'PPI_df': PPI_df,
        'tsne_paths': tsne_paths,
        'tp_tsne_paths': tp_tsne_paths,
        'bayes_factor_results': bayes_factor_results,
        'heatmap_paths': heatmap_paths,
        'condition_column': condition_column,
        'cell_type_column': cell_type_column
    }
    
    print("Evaluation completed successfully")
    return results