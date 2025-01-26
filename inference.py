import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import evaluate
import torch
import os 
import numpy as np
from utils import get_json_file
from data_processing import get_splitted_data, load_mlb_mapping
def get_test_data(inference_param_path = 'inference_params.json'):

  inference_params = get_json_file(inference_param_path)

  data_path = inference_params.get('data_path')
  train_ratio = inference_params['train_ratio']


  train_full_df, test_full_df, val_full_df = get_splitted_data(data_path, train_ratio)

  # get columns needed for model training
  test_data = test_full_df[['Title', 'Encoded Labels']]
  return test_data


def get_model(inference_param_path = 'inference_params.json'):
    inference_params = get_json_file(inference_param_path)
    model_saving_dir = inference_params.get('saved_model_dir')
    best_model_dir = inference_params.get('best_checkpoint_dir')
    trained_model_folder = os.path.join(model_saving_dir, best_model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(trained_model_folder)
    return model


def get_tokenizer(inference_param_path = 'inference_params.json'):
    inference_params = get_json_file(inference_param_path)  
    model_saving_dir = inference_params.get('saved_model_dir')
    tokenizer = AutoTokenizer.from_pretrained(model_saving_dir)
    return tokenizer
    
def get_classes():
  return  load_mlb_mapping()


def sigmoid(x):
   return 1/(1 + np.exp(-x))


def compute_inference_metrics(test_data, tokenizer, model, threshold=0.5):
  clf_metrics = evaluate.combine(["accuracy", "f1", "precision", "recall"])


  # Set model to evaluation mode
  model.eval()

  texts = list(test_data['Title'].values)
  labels = test_data['Encoded Labels']

  # Tokenize the texts
  inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt", max_length=512)

  # Perform inference
  with torch.no_grad():
      outputs = model(**inputs)
      logits = outputs.logits  # Raw scores
      predictions = sigmoid(logits)
      transformed_pred = (predictions.detach().cpu().numpy() > threshold).astype(int).reshape(-1)
      results = clf_metrics.compute(predictions=transformed_pred, references=labels.explode().values)
  return results


def divide_data_by_labels(data: pd.DataFrame, classes,label_column: str = 'Encoded Labels'):
    """Divides the data into classes based on encoded labels.
    Args:
        data: The input DataFrame containing the data and labels.
        label_column: The name of the column containing the encoded labels.

    Returns:
        A dictionary where keys represent the classes and values are the corresponding data slices.
    """
    class_encodings = np.unique(data[label_column].values)

      # Convert the label column to tuples for proper comparison
    data['Encoded Labels Tuple'] = data[label_column].apply(tuple)

    # Convert class_encodings to tuples
    class_encodings = [tuple(label_list) for label_list in class_encodings]

    # Iterate through class_encodings and filter data
    divided_data = {}
    for idx, class_label in enumerate(class_encodings):
        divided_data[idx] = data[data['Encoded Labels Tuple'] == class_label]
    
    return divided_data
