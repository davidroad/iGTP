# iGTP

**iGTP** is a Python-based package designed for gene expression analysis. Deep-learning models like variational autoencoder have enabled low dimensional cellular embedding representation for large-scale single-cell transcriptomes and shown great flexibility in downstream tasks. However, biologically meaningful latent space is usually missing if no specific structure is designed. Here, we engineered a novel interpretable generative transcriptional program (iGTP) framework that could model the importance of transcriptional program (TP) space and protein-protein interactions (PPIs) between different biological states. We demonstrate the performance of iGTP in a diverse biological context using Gene Ontology, canonical pathway, and different PPI curation. iGTP not only elucidated the ground truth of cellular responses but also surpassed other deep learning models and traditional bioinformatics methods in functional enrichment tasks. By integrating the latent layer with a graph neural network (GNN) framework, iGTP effectively inferred cellular responses to perturbations. We anticipate that iGTP will offer insights at both PPI and TP levels, and holds promise for predicting responses to novel perturbations.

![image](https://github.com/user-attachments/assets/02da9cdb-4847-44e8-b2b9-31d505403485)


---

## 🚀 Installation

### Option 1: Install via GitHub

1. Clone the repository:

   ```bash
   git clone https://github.com/davidroad/iGTP.git
   cd iGTP
   ```

2. Install the required dependencies:

   ```bash
   pip install -e .
   ```

3. Alternatively, you can install via pip directly from GitHub:

   ```bash
   pip install git+https://github.com/davidroad/iGTP.git
   ```

### Option 2: Using Docker (Recommended for Easy Setup)

You can use the provided Docker image to run the analysis without worrying about environment setup.

1. Pull the Docker image:

   ```bash
   docker pull freshnemo/test_docker:mda_igtp
   ```

2. Run the Docker container:

   ```bash
   docker run -v /path/to/your/data:/data -it freshnemo/test_docker:mda_igtp
   ```

   Replace `/path/to/your/data` with the actual path to your dataset.

---

## 📂 Directory Structure

```plaintext
iGTP/
├── iGTP/                     ← Main package code
│   ├── __init__.py           ← Python package initializer
│   ├── data/                 ← Data folder 
│   ├── model/                ← Model-related code (i.e., iGTP_Linear, iGTP_model)
│   ├── iGTP_analyze_tool.py  ← Optional tool for analysis
│   ├── iGTP_eval.py          ← Evaluation script
│   ├── iGTP_Kfold_train.py   ← Main KFold training script
│   ├── learning_utilities.py ← Utility functions for learning tasks
│   └── preprocess.py         ← Data preprocessing
├── setup.py                  ← Setup script for package installation
├── README.md                 ← Project documentation
└── requirements.txt          ← Python dependencies
```

---

## 🧑‍💻 Usage

To train the model, you need to run the `iGTP_Kfold_train.py` script. Here’s an example of how to run it:

1. First, make sure you have a configuration YAML file that defines parameters such as data file paths and model settings.

2. Run the script with the path to the YAML configuration:

   ```bash
   python iGTP/iGTP_Kfold_train.py --config /path/to/your/config.yaml
   ```
3. The above code will start training iGTP on the provided example data. You may replace the example data with your own data.
 
### Configuration Example

The training config file and evaluation config file are provided, including the example data location and the hyperparameters:

| Parameter                    | Value          | Description                                     |
| ---------------------------- | -------------- | ----------------------------------------------- |
| `tp_overlap_fraction`        | 0.1            | Fractional overlap allowed between TPs          |
| `encoder_layer_list`         | \[1500]        | List of units in encoder layers                 |
| `beta`                       | 0.00005        | Weight of KL divergence in VAE                  |
| `encoder_normal`             | 'batch'        | Normalization method in encoder (`batch`, etc.) |
| `init_type`                  | 'pos\_uniform' | Weight initialization strategy                  |
| `recon_loss`                 | 'mse'          | Reconstruction loss function                    |
| `drop_out`                   | 0.2            | Dropout rate                                    |
| `eps`                        | 0.001          | Small value to avoid numerical instability      |
| `cv_fold`                    | 3              | Number of folds in K-fold CV                    |
| `learning_rate`              | 0.001          | Learning rate                                   |
| `learning_rate_weight_decay` | 0.000054       | Weight decay for optimizer                      |
| `n_epochs`                   | 500            | Number of training epochs                       |
| `train_patience`             | 10             | Patience for early stopping on training         |
| `test_patience`              | 10             | Patience for early stopping on validation       |
| `data_num_workers`           | 1              | Number of workers for data loading              |
| `save_fold`                  | true           | Save model at each fold                         |
| `save_best`                  | true           | Save the best-performing model only             |
| `anneal_start`               | null           | Start step of KL annealing (optional)           |
| `anneal_time`                | null           | Duration of KL annealing (optional)             |
| `vb_nu`                      | 15000          | Variational Bayes parameter                     |
| `batch_size`                 | 4000           | Training batch size                             |
| `drop_last_batch`            | false          | Drop the last incomplete batch if needed        |
| `device_nu`                  | 0              | GPU device ID                                   |
| `using_gpu`                  | true           | Whether to use GPU                              |
| `z_sample`                   | 1              | Number of latent samples                        |


Example data are provided in the `data` folder. An additional example data file needs to be downloaded into the `data` folder to run the training code:
[Download example data from Google Drive](https://drive.google.com/file/d/1nPcup_W_xR7tkwgycHMqoz1d293vkPNr/view?usp=drive_link)


---

## 🛠️ Requirements

This package requires the following Python packages:

* **scanpy**: For scRNA-seq data analysis.
* **pyyaml**: For reading and writing configuration files.

To install the dependencies, you can use:

```bash
pip install -r requirements.txt
```

---

## ⚙️ Running the Model

1. **Load and Preprocess Data**: The data will be preprocessed using functions from `preprocess.py` before being fed into the model.

2. **Model Training**: The `KFoldTorch` model is used for K-fold cross-validation training. The training loop will be run from the `iGTP_Kfold_train.py` script.

---

## 🐳 Docker Usage

For a simple setup with all dependencies included, you can use the Docker image. To run the Docker container, you can map your local data to the Docker container:

```bash
docker run -v /path/to/your/data:/data -it freshnemo/test_docker:mda_igtp
```

This allows you to run the code in an isolated environment, ensuring you have the correct versions of dependencies.

---

## 🤝 Contributing

We welcome contributions! If you would like to contribute to this project:

1. Fork the repository
2. Create a feature branch
3. Submit a pull request

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 📑 Citation

If you use this package or the underlying model in your research, please cite the following paper:

**Learning interpretable cellular embedding for inferring biological mechanisms underlying single-cell transcriptomics**
[PubMed](https://pubmed.ncbi.nlm.nih.gov/39649598/)

---
