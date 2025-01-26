import numpy
import json
import matplotlib.pyplot as plt
import torch
import numpy as np

def get_json_file(json_path):
    with open(json_path, 'r') as file:
        data = json.load(file)
    return data



def count_and_visualize_labels(labels_series):
    """
    Counts the occurrences of each label in a Pandas Series and visualizes the results,
    including NaN values.

    Args:
        labels_series: A Pandas Series containing the labels.

    Returns:
        None (displays the visualization)
    """
    label_counts = labels_series.value_counts(dropna=True)

    # Create a bar chart
    plt.figure(figsize=(10, 6))  # Adjust figure size as needed
    label_counts.plot(kind='bar', color='skyblue')
    plt.title('Label Occurrences')
    plt.xlabel('Labels')
    plt.ylabel('Count')
    plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for readability
    plt.tight_layout()  # Adjust layout to prevent labels from overlapping
    plt.show()


def compute_class_weights(train_data, clamp_min = 0.02, clamp_max = 20):
    class_frequencies = train_data['Combined Labels'].explode().value_counts(normalize=False)

    # Total number of samples
    total_samples = class_frequencies.sum()

    # Compute weights: Inverse of the frequency normalized
    class_weights = total_samples / (len(class_frequencies) * class_frequencies)
    
    class_weights_tensor = torch.tensor(class_weights.values).float()
    # Convert to PyTorch tensor
    class_weights_tensor = torch.clamp(class_weights_tensor, min = clamp_min, max=clamp_max)
    return class_weights_tensor




def recover_labels_multi(one_hot_labels, classes):
    """
    Recover label names from one-hot encoded values for multi-label data.

    Args:
        one_hot_labels (np.ndarray): One-hot encoded array of shape (n_samples, n_classes).
        classes (list): List of class names corresponding to one-hot encoding.

    Returns:
        list: List of label names (as lists) corresponding to each row in `one_hot_labels`.
    """
    # Ensure the input is a NumPy array
    one_hot_labels = np.array(one_hot_labels)
    
    # Find indices of all 1s in each row
    label_indices = [np.where(row == 1)[0] for row in one_hot_labels]
    
    # Map indices to class names
    label_names = [[classes[idx] for idx in indices] for indices in label_indices]
    
    return label_names