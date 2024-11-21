# iGTP
Learning interpretable cellular embedding for inferring biological mechanisms underlying single-cell transcriptomics 
![image](https://github.com/user-attachments/assets/02da9cdb-4847-44e8-b2b9-31d505403485)

**Overview of iGTP framework**

Deep-learning models like variational autoencoder have enabled low dimensional cellular embedding representation for large-scale single-cell transcriptomes and shown great flexibility in downstream tasks. However, biologically meaningful latent space is usually missing if no specific structure is designed. Here, we engineered a novel interpretable generative transcriptional program (iGTP) framework that could model the importance of transcriptional program (TP) space and protein-protein interactions (PPIs) between different biological states. We demonstrate the performance of iGTP in a diverse biological context using Gene Ontology, canonical pathway, and different PPI curation. iGTP not only elucidated the ground truth of cellular responses but also surpassed other deep learning models and traditional bioinformatics methods in functional enrichment tasks. By integrating the latent layer with a graph neural network (GNN) framework, iGTP effectively inferred cellular responses to perturbations. We anticipate that iGTP will offer insights at both PPI and TP levels, and holds promise for predicting responses to novel perturbations.
