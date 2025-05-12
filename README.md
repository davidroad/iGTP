# iGTP
Learning interpretable cellular embedding for inferring biological mechanisms underlying single-cell transcriptomics 
https://pubmed.ncbi.nlm.nih.gov/39649598/
![image](https://github.com/user-attachments/assets/02da9cdb-4847-44e8-b2b9-31d505403485)

---

# iGTP

**iGTP** is a Python-based package designed for gene expression analysis. Deep-learning models like variational autoencoder have enabled low dimensional cellular embedding representation for large-scale single-cell transcriptomes and shown great flexibility in downstream tasks. However, biologically meaningful latent space is usually missing if no specific structure is designed. Here, we engineered a novel interpretable generative transcriptional program (iGTP) framework that could model the importance of transcriptional program (TP) space and protein-protein interactions (PPIs) between different biological states. We demonstrate the performance of iGTP in a diverse biological context using Gene Ontology, canonical pathway, and different PPI curation. iGTP not only elucidated the ground truth of cellular responses but also surpassed other deep learning models and traditional bioinformatics methods in functional enrichment tasks. By integrating the latent layer with a graph neural network (GNN) framework, iGTP effectively inferred cellular responses to perturbations. We anticipate that iGTP will offer insights at both PPI and TP levels, and holds promise for predicting responses to novel perturbations.


---

## 🚀 Installation

### Option 1: Install via GitHub

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/iGTP.git
   cd iGTP
   ```

2. Install the required dependencies:

   ```bash
   pip install -e .
   ```

3. Alternatively, you can install via pip directly from GitHub:

   ```bash
   pip install git+https://github.com/your-username/iGTP.git
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
│   ├── data/                 ← Data folder (if applicable)
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

### Configuration Example

Here’s an example of what the `config.yaml` file might look like:

```yaml
sc_data_path: "/path/to/your/data.h5ad"
tp_file_path: "/path/to/your/tf.gmt"
ppi_file_path: "/path/to/your/protein_interaction.txt"
recon_loss: "mse"
init_type: "xavier"
vb_nu: 0.01
```

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
