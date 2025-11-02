import pandas as pd
import numpy as np
import json
import sys

from normalize import *
from imputation import *


def sigmoid(z):
    """Sigmoid activation function"""
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))


def predict_probabilities(X, weights):
    """
    Predict probabilities for all classes using one-vs-all weights

    Args:
        X: Feature matrix (m x n)
        weights: Weight matrix (num_classes x (n+1)) including bias

    Returns:
        Probability matrix (m x num_classes)
    """
    m = X.shape[0]

    # Add bias term (column of ones)
    X_b = np.c_[np.ones((m, 1)), X]

    # Calculate probabilities for each class
    # weights shape: (num_classes, n+1)
    # X_b shape: (m, n+1)
    # Result shape: (m, num_classes)
    logits = X_b @ weights.T
    probabilities = sigmoid(logits)

    return probabilities


def predict_houses(probabilities, house_mapping):
    """
    Convert probability predictions to house names

    Args:
        probabilities: Matrix of probabilities (m x num_classes)
        house_mapping: Dictionary mapping house names to numeric labels

    Returns:
        Array of predicted house names
    """
    # Get the class with highest probability for each sample
    # +1 because labels are 1-4
    predicted_labels = np.argmax(probabilities, axis=1) + 1

    # Create reverse mapping (number -> house name)
    reverse_mapping = {v: k for k, v in house_mapping.items()}

    # Convert numeric predictions to house names
    predictions = [reverse_mapping[label] for label in predicted_labels]

    return predictions


def save_predictions(predictions, output_file='houses.csv'):
    """
    Save predictions to CSV file in the required format

    Args:
        predictions: Array of predicted house names
        output_file: Output CSV filename
    """
    df_output = pd.DataFrame({
        'Index': range(len(predictions)),
        'Hogwarts House': predictions
    })

    df_output.to_csv(output_file, index=False)
    print(f"Predictions saved to {output_file}")


def main():
    try:
        # Load dataset
        data_pred = pd.read_csv(sys.argv[1], index_col=0)
        with open(sys.argv[2], 'r') as f:
            params = json.load(f)
    except FileNotFoundError as e:
        print(f"Error: file not found: {e}")
        return False
    except IndexError:
        print("Error: Please provide 2 files as arguments")
        print("Usage: python logreg_predict.py <dataset.csv> <weights.json>")
        return False
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in {sys.argv[2]}")
        return False

    # Extract weights and feature information from saved parameters
    weights = np.array(params['theta'])
    feature_names = params['feature_names']
    house_mapping = params['house_mapping']

    print(f"Loaded model with {params['num_labels']} classes")
    print(f"Weight matrix shape: {weights.shape}")
    print(f"Features used: {feature_names}")

    # Normalize Data (same preprocessing as training)
    fill_data_correlation(data_pred)

    # Select the same features used in training
    try:
        new_data = data_pred[feature_names]
    except KeyError as e:
        print(f"Error: Missing required feature in dataset: {e}")
        return False

    # Normalize features
    df_norm = new_data.apply(normalize, axis=0)

    # Impute missing values
    df_norm_imputed = impute_missing_values(df_norm, k=5)

    # Verify no missing values
    if df_norm_imputed.isnull().sum().sum() > 0:
        print("Warning: There are still missing values after imputation")
        print(df_norm_imputed.isnull().sum())

    print(f"Dataset dimensions: {df_norm_imputed.shape}")

    # Prepare data for prediction
    X_test = df_norm_imputed.values

    # Make predictions
    print("\nMaking predictions...")
    probabilities = predict_probabilities(X_test, weights)
    predictions = predict_houses(probabilities, house_mapping)

    # Save predictions
    output_file = 'houses.csv'
    save_predictions(predictions, output_file)

    # Show sample predictions
    print(f"\nFirst 5 predictions:")
    for i in range(min(5, len(predictions))):
        print(
            f"Student {i}: {predictions[i]} (confidence: {probabilities[i].max():.3f})")

    print(f"\nTotal predictions made: {len(predictions)}")

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
