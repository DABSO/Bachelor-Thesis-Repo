# Bachelor Thesis: Daniel Neu

## Features

- **Data Loading & Preprocessing:** Efficiently load and preprocess input datasets.
- **Inference Pipeline:** Predict using machine learning models with flexibility to run locally or via Slurm.
- **Aggregation:** Aggregate model predictions into final results using either voting methods or ML-based aggregators
- **Evaluation:** Assess prediction performance against ground truth.
- **Baselines:** Calculate baseline predictions for comparative analysis.
- **Configuration Management:** Manage models, datasets, and prompts through config files.

## Setup

1. **Set Up the Environment**

   Ensure you have [Anaconda](https://www.anaconda.com/) installed. Then create and activate a Conda environment:

   ```bash
   conda env create -f environment.yml
   conda activate ba
   ```

2. **Install Dependencies**

  Install dependencies manually:

   ```bash
   pip install -r requirements.txt
   ```

3. **Configure Environment Variables**

   Create a `.env` file in the root directory and set the required environment variables specified in .env.example .

   ```env
   MODEL_DIR=/path/to/models
   HUGGINGFACE_TOKEN=your_huggingface_token
   ...
   ```

## Configuration

The project utilizes configuration files to manage models, prompts, and datasets. These are essential to tailor the repo to a specific scenario


## Usage
0. Setup the repo, such as .env, datasets, prompts, models, corresponding configs evaluate re-annotated data
1. Calculate the model outputs using run_inference.py
2. Aggregate the individual predictions using the aggregation scripts
3. Evaluate the aggregated predictions using the evaluation scripts  


