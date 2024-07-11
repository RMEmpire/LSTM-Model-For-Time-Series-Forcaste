# LSTM Stock Price Analysis

## Overview

This repository contains an LSTM model for analyzing and predicting stock prices. The project is organized into several directories and files that handle data preprocessing, model training, evaluation, and prediction.

## Repository Structure

- **Data/**: Contains all the CSV datasets used for training and testing the model.
- **Core/**:
  - `data.py`: Handles data preprocessing tasks, including reading CSV files, cleaning data, and preparing it for model training.
  - `model.py`: Contains the LSTM model architecture, training routines, and evaluation functions.
- **PreTrainedModel/**: Stores all the trained models.
- `run.py`: The main script that orchestrates the data preprocessing, model training, and evaluation by calling functions from `data.py` and `model.py`.

## Getting Started

### Prerequisites

Ensure you have the following packages installed:

- Python 3.x
- TensorFlow
- NumPy
- Pandas
- Scikit-learn

You can install the required packages using the following command:

```bash
pip install tensorflow numpy pandas scikit-learn
