# Gut Microbiome Food Allergy Prediction

A machine learning pipeline for predicting food allergy development from gut microbiome data using foundation models.

## Overview

This project uses a two-stage architecture to predict food allergies from microbiome samples:

1. **Embedding Generation**: Convert microbiome data into numerical representations using:
   - ProkBERT for DNA sequence embeddings
   - MicrobiomeTransformer for sample-level embeddings

2. **Classification**: Train classifiers (Logistic Regression, Random Forest, SVM, MLP) on the embeddings to predict allergic vs healthy subjects.

The pipeline includes automatic embedding generation, hyperparameter tuning, cross-validation, and comprehensive evaluation metrics.

## Scientific Background

Research shows that gut microbiome composition differs significantly between allergic and healthy children:
- Reduced microbial diversity in allergic subjects
- Depletion of protective taxa (Bifidobacterium, Faecalibacterium)
- Enrichment of pro-inflammatory bacteria (Escherichia-Shigella)
- Changes detectable months before clinical symptoms appear

This predictive potential enables early intervention strategies.

## Prerequisites

- Python 3.8 or higher
- 8GB+ RAM (16GB recommended)
- Optional: GPU for faster embedding generation (5-10x speedup)

## Installation

### Clone the Repository

```bash
git clone https://github.com/AI-For-Food-Allergies/gut_microbiome_project.git
cd gut_microbiome_project
```

### Install Dependencies

**Option 1: Using uv (recommended)**
```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync

# Activate environment
source .venv/bin/activate
```

**Option 2: Using pip**
```bash
pip install -e .
```

## Data Setup

You need three types of files to run the pipeline:

### 1. Dataset Files (Required)

Sample metadata in CSV format with columns: `sid` (sample ID) and `label` (0=healthy, 1=allergic)

**Download**: [Google Drive - Datasets](https://drive.google.com/drive/folders/1-MM3xOOhaEgILnD-D9IiLBrSBQOlz6QP?usp=sharing)

Available datasets:
- Tanaka
- Goldberg
- Diabimmune
- Gadir

**Location**: `data_preprocessing/datasets/<dataset_name>/`

Example:
```
data_preprocessing/
└── datasets/
    ├── diabimmune/
    │   ├── Month_1.csv
    │   └── Month_2.csv
    └── goldberg/
        ├── T1.csv
        └── T2.csv
```

### 2. Model Checkpoint (Required)

Pre-trained MicrobiomeTransformer model

**Download**: [Google Drive - Model Checkpoint](https://drive.google.com/file/d/1hykscEI4CbQm5ZzPOy9-o4HYjdwHL4u0/view?usp=sharing)

**File**: `checkpoint_epoch_0_final_epoch3_conf00.pt`

**Location**: `data/checkpoint_epoch_0_final_epoch3_conf00.pt`

### 3. OTU Mapping Files (Required)

Parquet files that map samples to OTUs to DNA sequences

**Download**: [Google Drive - Parquet Files](https://drive.google.com/drive/folders/1d33c5JtZREoDWRAu14o-fDXOpuriuyQC?usp=sharing)

**Files**:
- `samples-otus-97.parquet`
- `otus_97_to_dna.parquet`

**Location**: `data_preprocessing/mapref_data/`

### Optional: Pre-generated Embeddings

Skip the embedding generation step (which takes hours) by downloading preprocessed embeddings:

**Download**: [Google Drive - Preprocessed Data](https://drive.google.com/drive/folders/1bP98QGr1uXIhb2eZnqk4VcrHEj1su2rL?usp=sharing)

Available files:
- `diabimmune_data.zip` (4.58 GB)
- `gadir_data.zip` (49.5 MB)
- `goldberg_data.zip` (142.3 MB)
- `tanaka_data.zip` (33.8 MB)

Each zip contains three folders: `dna_sequences/`, `dna_embeddings/`, `microbiome_embeddings/`

Place the extracted folders in the corresponding dataset directory:
```bash
unzip diabimmune_data.zip
mv diabimmune_data/dna_sequences/* data_preprocessing/dna_sequences/diabimmune/
mv diabimmune_data/dna_embeddings/* data_preprocessing/dna_embeddings/diabimmune/
mv diabimmune_data/microbiome_embeddings/* data_preprocessing/microbiome_embeddings/diabimmune/
```

### Final Directory Structure

After setup:
```
gut_microbiome_project/
├── data/
│   └── checkpoint_epoch_0_final_epoch3_conf00.pt
├── data_preprocessing/
│   ├── datasets/
│   │   ├── diabimmune/
│   │   │   ├── Month_1.csv
│   │   │   └── Month_2.csv
│   │   └── goldberg/
│   │       ├── T1.csv
│   │       └── T2.csv
│   ├── mapref_data/
│   │   ├── samples-otus-97.parquet
│   │   └── otus_97_to_dna.parquet
│   ├── dna_sequences/        # Generated or downloaded
│   ├── dna_embeddings/       # Generated or downloaded
│   └── microbiome_embeddings/  # Generated or downloaded
└── config.yaml
```

## Configuration

Edit `config.yaml` to set your experiment parameters:

```yaml
data:
  dataset_path: "data_preprocessing/datasets/diabimmune/Month_2.csv"
  mirobiome_transformer_checkpoint: "data/checkpoint_epoch_0_final_epoch3_conf00.pt"
  device: "cpu"  # Options: "cpu", "cuda", "mps"

model:
  classifier: "logreg"  # Options: "logreg", "rf", "svm", "mlp"
  use_scaler: true

evaluation:
  cv_folds: 5
  results_output_dir: "eval_results"
```

See `config.yaml` for all available options including hyperparameter grids.

## Quick Start

### Run with Pre-generated Embeddings

If you downloaded pre-generated embeddings:

```bash
python main.py
```

The pipeline will load embeddings and train a classifier. Results are saved to `eval_results/`.

### Run with Automatic Embedding Generation

If you haven't pre-generated embeddings, the pipeline will create them automatically:

```bash
python main.py
```

**Note**: First run will be slow (hours) as embeddings are generated. Subsequent runs use cached embeddings.

## Usage

### Basic Evaluation

Run a single classifier with default parameters:

```python
# In main.py
run_evaluation(config)
```

```bash
python main.py
```

### Compare Multiple Classifiers

Evaluate and compare different algorithms:

```python
# In main.py
run_evaluation(config, classifiers=["logreg", "rf", "svm", "mlp"])
```

### Hyperparameter Tuning (Recommended)

Perform grid search with unbiased evaluation:

```python
# In main.py
run_grid_search_experiment(config, classifiers=["logreg", "rf", "svm", "mlp"])
```

This uses nested cross-validation:
1. Inner loop: Grid search to find best hyperparameters (5-fold CV, random_state=42)
2. Outer loop: Evaluate best model on fresh data (5-fold CV, random_state=123)

### Custom Hyperparameters

Override config with custom parameter grids:

```python
custom_grids = {
    "logreg": {"C": [0.01, 0.1, 1], "penalty": ["l2"]},
    "rf": {"n_estimators": [100, 200], "max_depth": [10, 20]}
}
run_grid_search_experiment(config, custom_param_grids=custom_grids)
```

## Understanding Results

Results are saved to `eval_results/<dataset_name>/<timestamp>/`:

```
eval_results/
└── diabimmune/
    └── Month_2/
        └── 2024-12-15_14-30-45/
            ├── Logistic_Regression/
            │   ├── classification_report.txt
            │   ├── confusion_matrix.png
            │   ├── confusion_matrix_normalized.png
            │   ├── roc_curve.png
            │   └── predictions.csv
            ├── Random_Forest/
            │   └── ...
            ├── combined_report.txt
            ├── comparison_roc_curves.png
            └── best_params_summary.json
```

**Files explained**:
- `classification_report.txt`: Precision, recall, F1-score per class
- `confusion_matrix.png`: True vs predicted labels (raw counts)
- `confusion_matrix_normalized.png`: Confusion matrix as percentages
- `roc_curve.png`: ROC curve with AUC score
- `predictions.csv`: Sample-level predictions with probabilities
- `combined_report.txt`: Comparison of all classifiers
- `comparison_roc_curves.png`: All ROC curves on one plot
- `best_params_summary.json`: Optimal hyperparameters found

## Advanced Topics

### Manual Embedding Generation

For more control over the embedding generation process, use the standalone script:

**1. Configure the script**:

Edit `generate_embeddings.py` (lines 202-203):
```python
BASE_OUTPUT_DIR = Path("data_preprocessing")
DATASET_DIR = Path("data_preprocessing/datasets/diabimmune")  # Update path
```

**2. Run the script**:
```bash
python generate_embeddings.py
```

**What gets generated**:
1. DNA sequences (CSV files, one per sample)
2. DNA embeddings (H5 format, ProkBERT embeddings)
3. Microbiome embeddings (H5 format, aggregated sample embeddings)

**Performance tips**:
- Use GPU: Set `DEVICE = "cuda"` in `generate_embeddings.py`
- You can interrupt (Ctrl+C) and resume - completed files are kept
- If interrupted during embedding generation, delete the incomplete H5 file before restarting

See `README_EMBEDDINGS.md` for detailed instructions on embedding generation.

### Programmatic API

**Load data**:
```python
from data_loading import load_dataset_df
from utils.data_utils import load_config, prepare_data

config = load_config()
dataset_df = load_dataset_df(config)  # Auto-generates embeddings if needed
X, y = prepare_data(dataset_df)  # X: embeddings, y: labels
```

**Use classifier**:
```python
from modules.classifier import SKClassifier

classifier = SKClassifier("logreg", config)

# Simple evaluation
metrics = classifier.evaluate_model(X, y, cv=5)
print(metrics.classification_report)

# Grid search with final evaluation
param_grid = {"C": [0.1, 1, 10], "penalty": ["l2"]}
metrics = classifier.grid_search_with_final_eval(
    X, y,
    param_grid=param_grid,
    grid_search_cv=5,
    final_eval_cv=5
)
```

### Leave-One-Dataset-Out Validation

To test cross-dataset generalization, train on multiple datasets and evaluate on a held-out dataset. This requires manual implementation combining datasets in the data loading pipeline.

## Troubleshooting

### Common Issues

**Out of memory**
- Reduce `batch_size_embedding` in `config.yaml`
- Use GPU/MPS for faster processing with less memory pressure

**Slow embedding generation**
- First run always takes hours (processing hundreds of samples)
- Use GPU: Set `device: "cuda"` or `device: "mps"` in config
- Download pre-generated embeddings instead

**Missing dependencies**
- Run `pip install -e .` or `uv sync`
- Activate virtual environment: `source .venv/bin/activate`

**Parquet file errors**
- Ensure both parquet files are in `data_preprocessing/mapref_data/`
- Verify files are not corrupted (re-download if needed)

**ModuleNotFoundError**
- Make sure you're in the project root directory
- Ensure virtual environment is activated
- Run installation command again

### Performance Tips

- Start with a small dataset to verify the pipeline works
- Use GPU/MPS for 5-10x faster embedding generation
- Pre-generate embeddings for multiple datasets in batch
- Embeddings are cached automatically - only generated once per dataset

## Project Structure

```
gut_microbiome_project/
├── config.yaml              # Configuration file
├── main.py                  # Main execution script
├── data_loading.py          # Data loading pipeline
├── generate_embeddings.py   # Standalone embedding generation
├── modules/                 # Core model classes
│   ├── model.py            # MicrobiomeTransformer
│   └── classifier.py       # SKClassifier wrapper
├── utils/                   # Helper utilities
│   ├── data_utils.py       # Data preparation
│   └── evaluation_utils.py # Results management
├── data_preprocessing/      # Data and preprocessing
│   ├── datasets/           # Dataset CSV files
│   ├── mapref_data/        # Parquet mapping files
│   └── ...                 # Generated embeddings
└── example_scripts/         # Usage examples
```

## Contributing

We welcome contributions from researchers, data scientists, and developers.

See [Contributing.md](Contributing.md) for:
- Development environment setup
- Code style guidelines
- Pull request process
- Testing requirements

To get involved, join the [huggingscience Discord server](https://discord.com/invite/VYkdEVjJ5J).

## Citation

If you use this code in your research, please cite:

```
[Citation information to be added]
```

## License

[License information to be added]

## Support

For questions or issues:
- Open an issue on GitHub
- Join the Discord server: https://discord.com/invite/VYkdEVjJ5J
- Check `README_EMBEDDINGS.md` for embedding generation details
