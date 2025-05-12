from iGTP_analyze_tool import *
import os
import argparse
import yaml
def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config
def main(ex_args):
    #  Load and preprocess data
    model,args, TP_emb, cell_gene ,gene_weight, cell_label_df,TP_df,_=load_and_evaluate_model(ex_args)
    # Plot t-SNE-species
    plot_tsne(cell_label_df, args, 'Species')
    # Plot t-SNE-Condiction
    plot_tsne(cell_label_df, args, 'Condiction')
    # Plot t-SNE-CellType
    plot_tsne(cell_label_df, args, 'Cell Type')
    # Plot t-SNE-TP-colored
    plot_tsne_emb_color(cell_label_df, args, TP_df, TP_emb)
    condiction_list=list(cell_label_df['Condiction'].unique())
    species_list=list(cell_label_df['Species'].unique())
    # plot Condiction based analyze
    for i in range(len(condiction_list)):
        new_cell_label_df=cell_label_df[cell_label_df['Condiction']==condiction_list[i]]
        new_TP_emb=TP_emb[new_cell_label_df.cell_index.values,]
        print(new_TP_emb.shape)
        new_cell_label_df['cell_index']=range(len(new_cell_label_df))
        print(len(new_cell_label_df))
        tsne=new_cell_label_df[['dim_1','dim_2']].values
        print(tsne.shape)
    
        for m in range(len(species_list)):
            for n in range(m+1, len(species_list)):
                compare_condiciton = [species_list[m], species_list[n]]
                _,_= compute_and_save_bayes_factor(model, args,new_TP_emb, new_cell_label_df, TP_df, 'Species', 'Cell Type', 
                                compare_condiciton, tsne, condiction_list[i],m_permutation=12000)
        output_path=os.path.join(args['output_folder'],condiction_list[i],'cluster_heatmap.pdf')
        create_cell_type_condition_heatmap(new_TP_emb, new_cell_label_df, 'Cell Type',TP_df, 'TP_index', 'TP_name',
                                            'Species', species_list, output_path=output_path)
    # plot Condiction based analyze
    for i in range(len(species_list)):
        new_cell_label_df=cell_label_df[cell_label_df['Species']==species_list[i]]
        new_TP_emb=TP_emb[new_cell_label_df.cell_index.values,]
        new_cell_label_df['cell_index']=range(len(new_cell_label_df))
        tsne=new_cell_label_df[['dim_1','dim_2']].values
        for m in range(len(condiction_list)):
            for n in range(m+1, len(condiction_list)):
                compare_condiciton = [condiction_list[m], condiction_list[n]]
                _,_= compute_and_save_bayes_factor(model, args,new_TP_emb, new_cell_label_df, TP_df, 'Condiction', 'Cell Type', 
                                compare_condiciton, tsne, species_list[i],m_permutation=12000)
        output_path=os.path.join(args['output_folder'],species_list[i],'cluster_heatmap.pdf')
        create_cell_type_condition_heatmap(new_TP_emb, new_cell_label_df, 'Cell Type',TP_df, 'TP_index', 'TP_name',
                                            'Condiction', condiction_list, output_path=output_path)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run IGTP KFold PBMC analysis")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML configuration file")
    
    args = parser.parse_args()
    
    # Load configuration from YAML file
    config = load_config(args.config)
    
    # Set model prefix
   
    main(config)

