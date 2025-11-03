# ML-Based Prediction of Food Allergy Development From Gut Microbiome Data

## Project Overview

This project aims to develop a robust machine learning classifier that can predict the development of food allergies by analyzing an individual's gut microbiome data. By leveraging the latest advancements in microbiome foundation models, we seek to transform the landscape of food allergy prediction and prevention.

The core goal is to build a **binary classifier** that can distinguish between "healthy" and "allergic" subjects based on their gut microbiota composition. Future iterations will explore extending this model to predict the risk of developing specific types of food allergies.

## Motivation: The Microbiome-Allergy Link

Accumulating scientific evidence highlights the critical role of the gut microbiota in shaping immune tolerance to dietary antigens. Key findings that motivate this research include:

*   **Reduced Diversity:** Allergic children consistently exhibit reduced microbial diversity compared to healthy controls.
*   **Taxa Shifts:** There is a notable depletion of protective taxa (e.g., *Bifidobacterium*, *Faecalibacterium*, butyrate-producing *Clostridia*) and an enrichment of pro-inflammatory bacteria (e.g., *Escherichia-Shigella*, *Ruminococcus gnavus*).
*   **Predictive Potential:** These microbial shifts are often detectable months before the clinical manifestation of food allergies, suggesting a strong predictive potential for early intervention.

## Methodology

Our approach is built on a modern, two-stage machine learning architecture designed to handle the complexity and high dimensionality of microbiome data:

1.  **Microbiome Embedding Model:** We utilize a recently developed **foundation model** for microbiome data. This model serves as a backbone, extracting rich, low-dimensional, and meaningful representations (embeddings) from raw 16S rRNA or shotgun metagenomics data.
2.  **Classifier Head:** The extracted embeddings are then fed into a simpler machine learning model, such as **Logistic Regression** or a similar classifier, to perform the final binary prediction.

### Evaluation Strategy

To ensure the model is robust and generalizable, we employ a rigorous evaluation strategy:

*   **Standard Metrics:** Performance is assessed using standard classification metrics (e.g., AUC, F1-score, Accuracy) with cross-validation.
*   **Cross-Dataset Validation:** We implement a **leave-one-dataset-out** validation strategy to test the model's ability to generalize across different cohorts, sequencing protocols, and study conditions, which is crucial for real-world applicability.

## Repository Structure

The project follows a modular structure to separate concerns and facilitate collaboration:

| Directory/File | Purpose |
| :--- | :--- |
| `data_preprocessing/` | Scripts for cleaning, transforming, and preparing raw microbiome data. |
| `modules/` | Reusable classes and functions for the model architecture (e.g., the `MicrobiomeTransformer` wrapper). |
| `evaluation/` | Scripts for model performance assessment, metric calculation, and visualization. |
| `train.py` | The main entry point for running the model training pipeline. |
| `main.py` | The overall execution script for the entire project workflow. |
| `data_loading.py` | Utility functions for handling data I/O and alignment. |

## Getting Started

To set up the project environment and begin contributing:

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/AI-For-Food-Allergies/gut_microbiome_project.git
    cd gut_microbiome_project
    ```
2.  **Install dependencies:**
    ```bash
    # Using pip
    pip install -e .
    
    # Or using uv (if installed)
    uv sync
    ```
3.  **Download the Microbiome Transformer checkpoint:**
    
    The pre-trained Microbiome Transformer model checkpoint can be downloaded from Figshare:
    
    [Model and Data for diabimmune example](https://figshare.com/articles/dataset/Model_and_Data_for_diabimmune_example/30429055?file=58993825)
    

## 🤝 Contributing

We welcome contributions from researchers, data scientists, and developers!

Please refer to our dedicated **[Contributing Guide](https://github.com/AI-For-Food-Allergies/gut_microbiome_project/blob/master/Contributing.md)** for detailed instructions on:

*   Setting up your development environment.
*   The current development roadmap and open issues.
*   Guidelines for submitting pull requests and writing clean code.

**Want to be assigned to an issue?** Join the [huggingscience Discord server](https://discord.com/invite/VYkdEVjJ5J) and communicate your interest in the relevant channel!

Thank you for helping us advance the prediction of food allergies!
