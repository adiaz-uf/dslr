import sys
import os
import pandas as pd  # type: ignore
import numpy as np  # type: ignore
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import json

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V3.Logistic_Regression'))

from normalize import fill_data_correlation, normalize
from imputation import impute_missing_values

# Train Variables
learning_rate = 0.1
num_epochs = 100
batch_size = 32
epsilon = 1e-8


# Logistic Regression Base Functions
def softmax(z):
    """
    Softmax function for multinomial classification

    Args:
        z: Matrix of logits (m x num_classes)

    Returns:
        Probabilities matrix (m x num_classes)
    """
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


def compute_cost(X, y, W):
    """
    Compute cross-entropy cost for multinomial logistic regression

    Args:
        X: Feature matrix (m x n)
        y: One-hot encoded labels (m x num_classes)
        W: Weight matrix (n x num_classes)

    Returns:
        Cost value
    """
    m = X.shape[0]
    z = np.dot(X, W)
    h = softmax(z)

    # Clip to prevent log(0)
    epsilon = 1e-15
    h = np.clip(h, epsilon, 1 - epsilon)

    # Cross-entropy cost
    cost = (-1 / m) * np.sum(y * np.log(h))
    return cost


def minibatch_gradient_descent(X, y, learning_rate=0.01, num_epochs=100, batch_size=32, epsilon=1e-8):
    """
    Mini-batch gradient descent for multinomial logistic regression

    Args:
        X: Feature matrix (including bias column)
        y: One-hot encoded label matrix
        learning_rate: Learning rate (alpha)
        num_epochs: Number of complete passes through the dataset
        batch_size: Size of each mini-batch
        epsilon: Threshold for early stopping

    Returns:
        W: Optimized weight matrix
        cost_history: List with cost values per epoch
    """
    # Initialize weight matrix with small random values
    n_features = X.shape[1]
    n_classes = y.shape[1]
    W = np.random.randn(n_features, n_classes) * 0.01

    # Total number of samples
    m = X.shape[0]

    # Calculate number of mini-batches per epoch
    n_batches = int(np.ceil(m / batch_size))

    # List to save cost history
    cost_history = []

    # Calculate initial cost
    current_cost = compute_cost(X, y, W)
    cost_history.append(current_cost)
    print(f'Initial cost: {current_cost}')

    # Training loop by epochs
    for epoch in range(num_epochs):
        # Shuffle data at the beginning of each epoch
        permutation = np.random.permutation(m)
        X_shuffled = X[permutation]
        y_shuffled = y[permutation]

        # Mini-batch training
        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = min(start_idx + batch_size, m)

            # Get current mini-batch
            X_batch = X_shuffled[start_idx:end_idx]
            y_batch = y_shuffled[start_idx:end_idx]

            # Calculate predictions for this mini-batch
            z_batch = np.dot(X_batch, W)
            h_batch = softmax(z_batch)

            # Calculate gradient for this mini-batch
            batch_size_actual = end_idx - start_idx
            gradient = (1/batch_size_actual) * \
                np.dot(X_batch.T, (h_batch - y_batch))

            # Update weights
            W = W - learning_rate * gradient

        # Calculate and save cost at the end of each epoch
        current_cost = compute_cost(X, y, W)
        cost_history.append(current_cost)

        # Print progress every 10 epochs
        if epoch % 10 == 0:
            print(f'Epoch {epoch}: Cost = {current_cost}')

        # Early stopping
        if epoch > 0 and abs(cost_history[-2] - current_cost) < epsilon:
            print(f'\nConvergence reached at epoch {epoch}')
            print(f'Cost difference: {abs(cost_history[-2] - current_cost)}')
            break

    return W, cost_history


def predict(X, W):
    """
    Make predictions using trained weights

    Args:
        X: Feature matrix
        W: Weight matrix

    Returns:
        Array of predicted class labels (1-indexed)
    """
    probabilities = softmax(np.dot(X, W))
    return np.argmax(probabilities, axis=1) + 1  # +1 for 1-indexed classes


def plot_cost_evolution(cost_history):
    """
    Plot the evolution of cost during training

    Args:
        cost_history: List of cost values
    """
    plt.figure(figsize=(10, 6))
    plt.plot(cost_history)
    plt.title('Cost Evolution (J) during Mini-Batch GD Training')
    plt.xlabel('Epochs')
    plt.ylabel('Cost')
    plt.grid(True)
    plt.show()


def main():
    try:
        # Load dataset
        data = pd.read_csv(sys.argv[1], index_col=0)
    except FileNotFoundError:
        print(f"ERROR: CSV file not found: {sys.argv[1]}")
        return
    except IndexError:
        print("ERROR: Please provide dataset file as argument")
        print("Usage: python logreg_train_minibatch.py <dataset.csv>")
        return

    # Extract labels (Hogwarts House)
    y_labels = data['Hogwarts House'].copy()

    # Normalize Data
    fill_data_correlation(data)
    fields = data.select_dtypes(include=['float64']).columns
    fields = fields.drop('Astronomy')
    fields = fields.drop('History of Magic')
    fields = fields.drop('Arithmancy')
    fields = fields.drop('Potions')
    fields = fields.drop('Care of Magical Creatures')
    print("\nSelected features:")
    print(fields)

    new_data = data[fields]

    # Normalize features
    df_norm = new_data.apply(normalize, axis=0)

    # Impute missing values
    complete_rows = df_norm.dropna()
    incomplete_rows = df_norm[df_norm.isnull().any(axis=1)]
    print(f"\nNumber of complete rows: {len(complete_rows)}")
    print(f"Number of incomplete rows: {len(incomplete_rows)}")

    df_norm_imputed = impute_missing_values(df_norm, k=5)

    print("\nVerify no missing values:")
    print(df_norm_imputed.isnull().sum())
    print(f"Dataset dimensions: {df_norm_imputed.shape}")

    # Prepare data for training
    X = df_norm_imputed.values

    # Encode labels to numbers (0-3 for one-hot encoding)
    house_mapping = {
        'Gryffindor': 0,
        'Hufflepuff': 1,
        'Ravenclaw': 2,
        'Slytherin': 3
    }

    # Map house names to numbers
    y_numeric = y_labels.map(house_mapping).values
    num_labels = len(house_mapping)

    # Create one-hot encoded labels
    y = np.zeros((len(y_numeric), num_labels))
    for i, label in enumerate(y_numeric):
        y[i, label] = 1

    # Add bias term
    X_b = np.c_[np.ones((X.shape[0], 1)), X]

    print(
        f"\nTraining with {num_labels} classes using Mini-Batch Gradient Descent")
    print(f"Features shape: {X_b.shape}")
    print(f"Labels shape: {y.shape}")
    print(f"Batch size: {batch_size}")
    print(f"Number of epochs: {num_epochs}")
    print(f"Learning rate: {learning_rate}")

    # Train model with Mini-Batch Gradient Descent
    print("\nStarting training...")
    W_optimal, cost_history = minibatch_gradient_descent(
        X_b,
        y,
        learning_rate=learning_rate,
        num_epochs=num_epochs,
        batch_size=batch_size,
        epsilon=epsilon
    )

    print(f"\nTraining complete!")
    print(f"Final cost: {cost_history[-1]:.6f}")

    # Plot cost evolution
    plot_cost_evolution(cost_history)

    # Save parameters to JSON
    # Convert back to 1-indexed for consistency with other training methods
    house_mapping_1indexed = {
        'Gryffindor': 1,
        'Hufflepuff': 2,
        'Ravenclaw': 3,
        'Slytherin': 4
    }

    params_to_save = {
        'technique': 'Mini-Batch Gradient Descent',
        'num_labels': num_labels,
        'theta': W_optimal.T.tolist(),  # Transpose to match one-vs-all format
        'feature_names': fields.tolist(),
        'house_mapping': house_mapping_1indexed,
        'hyperparameters': {
            'learning_rate': learning_rate,
            'num_epochs': num_epochs,
            'batch_size': batch_size
        }
    }

    output_file = 'trained_params.json'
    with open(output_file, 'w') as f:
        json.dump(params_to_save, f, indent=4)

    print(f"\nSuccess: Parameters saved in '{output_file}'")
    print(
        f"Dimensions of saved theta (Classes x coefficients): {W_optimal.T.shape}")


if __name__ == "__main__":
    main()
