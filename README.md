# Tabular Data Regression Project

This project implements a regression model to predict a target based on 53 anonymized features.

## Project Setup

1. Clone the repository:
   ```
   git clone https://github.com/yuragorlo/comm_it.git
   cd comm_it
   ```

2. Create and activate a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Project Structure

- `dataset/`: Contains the input data files (train.csv and hidden_test.csv)
- `src/`: Contains the source code
  - `train.py`: Script for model training
  - `predict.py`: Script for model inference on test data
- `requirements.txt`: List of required Python packages
- `README.md`: This file

## Usage

1. To train the model:
   ```
   python src/train.py
   ```
   This will train the model and display the RMSE for both training and validation sets.

2. To generate predictions:
   ```
   python src/predict.py
   ```
   This will create a file `predictions.csv` in the `dataset/` directory with the predictions for the hidden test set.

## Model Details

The model uses a pipeline with the following steps:
1. Preprocessing (StandardScaler and MinMaxScaler)
2. Polynomial Features
3. Correlation Filter
4. PCA
5. LightGBM Regressor

The hyperparameters for the LightGBM model were determined through a randomized search and are set in the `train.py` file.
