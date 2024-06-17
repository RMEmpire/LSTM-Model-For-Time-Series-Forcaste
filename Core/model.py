import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import pickle

class StockPricePredictor:
    def __init__(self, dataframe, scaler):
        self.dataframe = dataframe
        self.scaler = scaler
        self.model = None

    def create_sequences(self, sequence_length=10, target_column='Close'):
        """Creates sequences for LSTM training."""
        sequences = []
        targets = []

        for i in range(len(self.dataframe) - sequence_length):
            sequence = self.dataframe.iloc[i : i + sequence_length][target_column]
            target = self.dataframe.iloc[i + sequence_length][target_column]
            sequences.append(sequence.values)
            targets.append(target)

        return np.array(sequences), np.array(targets)

    def build_model(self, sequence_length, feature_dim=1):
        """Builds the LSTM model."""
        self.model = Sequential([
            LSTM(50, activation='relu', input_shape=(sequence_length, feature_dim)),
            Dense(1)
        ])
        self.model.compile(optimizer='adam', loss='mse')

    @staticmethod
    def prepare_data(sequences, targets, sequence_length, test_size=0.2, shuffle=False):
        global X_train, X_test, y_train, y_test
        X_train, X_test, y_train, y_test = train_test_split(sequences, targets, test_size=test_size, shuffle=shuffle)
        # Reshape the training and testing input sequences
        X_train = X_train.reshape((len(X_train), sequence_length, 1))
        X_test = X_test.reshape((len(X_test), sequence_length, 1))

    def train_model(self, epochs=50, batch_size=32):
        """Trains the LSTM model."""
        global X_train, y_train
        history = self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=2)
        return history

    def evaluate_model(self, X_test, y_test):
        """Evaluates the model on test data."""
        y_pred = self.model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        rmse = mse ** 0.5
        mae = self.calculate_mae(y_test, y_pred)
        r2 = self.calculate_r2_score(y_test, y_pred)
        print(f"Mean Squared Error (MSE): {mse}")
        print(f"Root Mean Squared Error (RMSE): {rmse}")
        print(f"Mean Absolute Error (MAE): {mae}")
        print(f"R² Score: {r2}")
        return mse, rmse, mae, r2

    @staticmethod
    def calculate_mae(y_true, y_pred):
        """Calculates the Mean Absolute Error (MAE)."""
        return mean_absolute_error(y_true, y_pred)

    @staticmethod
    def calculate_r2_score(y_true, y_pred):
        """Calculates the R² score."""
        return r2_score(y_true, y_pred)

    def save_model(self, file_name: str):
        """Saves the trained model to a file."""
        with open(file_name, 'wb') as file:
            pickle.dump(self.model, file)
        print(f"Model saved to {file_name}.")

    def load_model(self, file_name: str):
        """Loads the model from a file."""
        if os.path.exists(file_name):
            with open(file_name, 'rb') as file:
                self.model = pickle.load(file)
            print(f"Model loaded from {file_name}.")
        else:
            print(f"Error: The file {file_name} does not exist.")

    def visualize(self, y_label):
        """Plots graph for stock values vs. date."""
        if 'Date' in self.dataframe.columns and y_label in self.dataframe.columns:
            self.dataframe['Date'] = pd.to_datetime(self.dataframe['Date'])
            plt.figure(figsize=(14, 7))
            sns.lineplot(data=self.dataframe, x='Date', y=y_label)
            plt.title(f'{y_label} Stock Values vs. Date')
            plt.xlabel('Date')
            plt.ylabel(f'{y_label} Stock Value')
            plt.show()
        else:
            print(f"Dataframe does not contain 'Date' and '{y_label}' columns.")

    def run_predictions_and_evaluation(self):
        """
        Performs predictions on the test data, evaluates the model, and validates the loaded model.

        Returns:
        dict: A dictionary containing predictions and evaluation metrics for both the original and loaded models.
        """
        global X_test, y_test

        # Perform predictions on the test data
        global y_pred
        y_pred = self.model.predict(X_test)

        # Evaluate the model
        mse, rmse, mae, r2 = self.evaluate_model(X_test, y_test)

        # Validate the loaded model using test data
        y_pred_loaded = self.model.predict(X_test)
        mse_loaded, rmse_loaded, mae_loaded, r2_loaded = self.evaluate_model(X_test, y_test)

        results = {
            'original_model': {
                'predictions': y_pred,
                'metrics': {
                    'mse': mse,
                    'rmse': rmse,
                    'mae': mae,
                    'r2': r2
                }
            },
            'loaded_model': {
                'predictions': y_pred_loaded,
                'metrics': {
                    'mse': mse_loaded,
                    'rmse': rmse_loaded,
                    'mae': mae_loaded,
                    'r2': r2_loaded
                }
            }
        }

        return results

    def visualize_predictions(self):
        """Plots actual vs. predicted values."""
        global y_test, y_pred
        y_pred = self.model.predict(X_test)
        plt.figure(figsize=(10, 6))
        plt.plot(y_test, label='Actual')
        plt.plot(y_pred, label='Predicted')
        plt.title('Actual vs. Predicted Stock Prices')
        plt.xlabel('Time')
        plt.ylabel('Stock Price')
        plt.legend()
        plt.show()


