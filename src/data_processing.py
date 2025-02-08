from sklearn.preprocessing import MultiLabelBinarizer
import pandas as pd
import pickle
import re
from sklearn.model_selection import train_test_split
import torch
from datasets import Dataset

def multi_label_binarization(df, column_name):
    """
    Encode multi-labels in the specified column of the DataFrame using MultiLabelBinarizer.

    Parameters:
    df (pandas.DataFrame): The input DataFrame.
    column_name (str): The column name containing the labels to be encoded.

    Returns:
    pandas.DataFrame: The DataFrame with an additional column for encoded labels.
    """
    # Initialize MultiLabelBinarizer
    mlb = MultiLabelBinarizer()
    
    # Fit and transform the labels
    y_encoded = mlb.fit_transform(df[column_name])
    
    # Display the encoded labels and classes
    print("Encoded labels:\n", y_encoded)
    print("\nClasses:\n", mlb.classes_)
    
    # Add the encoded labels as a new column in the DataFrame
    df['Encoded Labels'] = y_encoded.tolist()
    
    return df, mlb.classes_, mlb




def save_mlb_mapping(mlb, file_path='mlb_mapping.pkl'):
    """
    Save the MultiLabelBinarizer mapping (the classes_) to a file.

    Parameters:
    mlb (MultiLabelBinarizer): The MultiLabelBinarizer instance.
    file_path (str): The path to the file where the mapping will be saved.
    """
    with open(file_path, 'wb') as f:
        pickle.dump(mlb.classes_, f)
    print(f"Mapping saved to {file_path}")
    

def load_mlb_mapping(file_path='mlb_mapping.pkl'):
    """
    Load the MultiLabelBinarizer mapping (the classes_) from a file.

    Parameters:
    file_path (str): The path to the file from which the mapping will be loaded.

    Returns:
    list: The loaded classes.
    """
    with open(file_path, 'rb') as f:
        loaded_classes = pickle.load(f)
    print(f"Mapping loaded from {file_path}")
    return loaded_classes


def clean_title(title):
    """
    The function intended for cleaning text of 'Title' column
    """
    if pd.isnull(title):
        return title  # Skip cleaning for NaN values
    # Replace separators with spaces
    title = re.sub(r'[;,/]', ' ', title)
    # Retain meaningful special characters
    title = re.sub(r'[^a-zA-Z0-9\s\+\-]', '', title)
    # Normalize whitespace
    title = re.sub(r'\s+', ' ', title).strip()
    return title

def make_stratify_splitting(data, train_size, random_state, stratify_column):
    """
        Performs a stratified train-test split while handling single-occurrence categories in the stratification column.

        The function identifies categories in the `stratify_column` that occur only once and excludes them from 
        stratification. After the stratified split, these single-occurrence cases are added back to the training set.

        Parameters:
        -----------
        data : pd.DataFrame
            The dataset to split. It must contain the `stratify_column` used for stratification.

        train_size : float
            The proportion of the dataset to include in the training set. Value should be between 0 and 1.

        random_state : int
            The random seed used for reproducibility of the split.

        stratify_column : str
            The name of the column in `data` used for stratification. Stratification ensures 
            the distribution of categories in this column is preserved between the training and test sets.


    """

    # Get indices of single-occurrence cases
    single_occurrence_indices = data[data[stratify_column].isin(
        data[stratify_column].value_counts()[data[stratify_column].value_counts() == 1].index
    )].index

    # Filter out those cases for stratification
    data_to_stratify = data[~data.index.isin(single_occurrence_indices)]

    # Ensure there is enough data for stratification
    if data_to_stratify.empty:
        raise ValueError("Not enough data for stratification after filtering single-occurrence cases.")

    # Perform stratified split
    train_data, test_data = train_test_split(
        data_to_stratify,
        train_size=train_size,
        random_state=random_state,
        stratify=data_to_stratify[stratify_column]
    )

    # Combine train data with single-occurrence cases
    train_full_data = pd.concat([train_data, data.loc[single_occurrence_indices]])

    return train_full_data, test_data


def create_class_mappings(classes):
    """
    Create mappings between class labels and their corresponding IDs.

    Args:
        classes (list): A list of class labels.

    Returns:
        tuple: A tuple containing:
            - class2id (dict): A dictionary mapping class labels to IDs.
            - id2class (dict): A dictionary mapping IDs to class labels.
    """
    class2id = {label: i for i, label in enumerate(classes)}
    id2class = {i: label for label, i in class2id.items()}
    return class2id, id2class



def tokenize_function(data, tokenizer):
    return tokenizer(data["Title"], padding="max_length", truncation=True, max_length=128)

def preprocess_labels(data):
    data['labels'] = torch.tensor(data['Encoded Labels'], dtype = torch.float)  # Multi-hot encoding
    return data

def tokenization_processings(data, tokenizer):
  """
  Function converts df to huggingface Dataset and applies tokenization to full dataset.
  """

  dataset = Dataset.from_pandas(data)

  tokenized_datasets = dataset.map(lambda data: tokenize_function(data, tokenizer), batched=True)
  tokenized_datasets = tokenized_datasets.map(preprocess_labels, batched=True)
  return tokenized_datasets

def delete_blank_labels(df):
    df = df[df['Combined Labels'].apply(lambda x: x != [])]
    return df


def get_initial_preprocessings(df):

    df['Combined Labels'] = df[['Column 1', 'Column 2', 'Column 3', 'Column 4']].apply(
    lambda row: [x for x in row if pd.notna(x)], axis=1
        )
    df = delete_blank_labels(df)
    df, _,_ = multi_label_binarization(df, 'Combined Labels')
    df['Title'] = df['Title'].apply(clean_title)

    return df 

def get_splitted_data(data_path, train_ratio):
    df = pd.read_csv(data_path)
    df = get_initial_preprocessings(df) 
    train_full_df, test_full_df = make_stratify_splitting(data= df, train_size = train_ratio, random_state = 42, stratify_column = 'Encoded Labels')
    val_full_df, test_full_df = make_stratify_splitting(data = test_full_df, train_size = 0.5, random_state = 42, stratify_column = 'Encoded Labels')
    return train_full_df, test_full_df, val_full_df
