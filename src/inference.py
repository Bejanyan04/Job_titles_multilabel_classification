import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import evaluate
import torch
import json
import numpy as np
from src.utils import get_json_file
from src.data_processing import get_splitted_data, load_mlb_mapping, clean_title
def get_test_data(inference_param_path = 'inference_params.json'):

  inference_params = get_json_file(inference_param_path)

  data_path = inference_params.get('data_path')
  train_ratio = inference_params['train_ratio']


  train_full_df, test_full_df, val_full_df = get_splitted_data(data_path, train_ratio)

  # get columns needed for model training
  test_data = test_full_df[['Title', 'Encoded Labels']]
  return test_data

    
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

def get_model(inference_param_path = 'inference_params.json'):
    inference_params = get_json_file(inference_param_path)
    model_repo_id = inference_params.get("trained_model_hub_repo_id")
    #model_path = inference_params.get("trained_model_path")
    model = AutoModelForSequenceClassification.from_pretrained(model_repo_id)
    return model


def get_tokenizer(inference_param_path = 'inference_params.json'):
    inference_params = get_json_file(inference_param_path) 
    model_repo_id = inference_params.get("trained_model_hub_repo_id")
    #model_path = inference_params.get("trained_model_path")
    tokenizer = AutoTokenizer.from_pretrained(model_repo_id)
    return tokenizer

def get_sample_prediction(input_data, model, tokenizer, threshold = 0.5):
  #preprocess input_data
  cleaned_text = clean_title(input_data)
  tokenized_data = tokenizer(cleaned_text, padding="max_length", truncation=True, max_length=128)
   
  model.eval()
  with torch.no_grad():
    outputs = model(tokenized_data)
    logits = outputs.logits  # Raw scores
    predictions = sigmoid(logits)
    transformed_pred = (predictions.detach().cpu().numpy() > threshold).astype(int).reshape(-1)
    return transformed_pred
    


def inference_pipeline(inference_params_path):
  #classes = load_mlb_mapping()
  test_data = get_test_data()
  tokenizer = get_tokenizer()
  model = get_model() #get best model( finetuned huggingface model)
  metrics  = compute_inference_metrics(test_data, tokenizer, model, threshold=0.5)
  # Define the JSON file name
  json_file_path = "metrics_results.json"

  # Save the dictionary as a JSON file
  with open(json_file_path, "w") as json_file:
      json.dump(metrics, json_file, indent=4)

  print(f"Metrics saved to {json_file_path}")

    
