from transformers import Trainer
import numpy as np
from torch.nn import BCEWithLogitsLoss
from transformers.integrations import TensorBoardCallback
import pandas as pd
from utils import get_json_file, compute_class_weights
from data_processing import  multi_label_binarization, get_splitted_data, create_class_mappings, tokenization_processings
from transformers import AutoTokenizer,  DataCollatorWithPadding, AutoModelForSequenceClassification, TrainingArguments

import argparse

import evaluate


def sigmoid(x):
   return 1/(1 + np.exp(-x))

def compute_metrics(eval_pred):
   clf_metrics = evaluate.combine(["accuracy", "f1", "precision", "recall"])
   predictions, labels = eval_pred
   predictions = sigmoid(predictions)
   predictions = (predictions > 0.5).astype(int).reshape(-1)
   return clf_metrics.compute(predictions=predictions, references=labels.astype(int).reshape(-1))

class WeightedLoss(BCEWithLogitsLoss):
    def __init__(self, weights):
        super().__init__(reduction='none')  # No reduction, apply weights manually
        self.weights = weights

    def forward(self, logits, labels):
        loss = super().forward(logits, labels)
        weighted_loss = loss * self.weights
        return weighted_loss.mean()


class CustomTrainer(Trainer):
    def __init__(self, weighted_loss=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.weighted_loss = weighted_loss
        
    def compute_loss(self, model, inputs, return_outputs=False,num_items_in_batch=None):
        """
        Override the compute_loss method to use the custom loss function.
        """
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.logits # Access logits from outputs
        loss = self.weighted_loss(logits, labels)  # Apply custom loss
        return (loss, outputs) if return_outputs else loss


def train_model(data_parameters_path = 'data_parameters.json', training_params_path = 'model_training_args.json'):
    general_params= get_json_file(data_parameters_path)
    training_params = get_json_file('model_training_args.json')
    train_ratio = training_params['train_ratio']
    data_path = general_params['data_path']

    train_full_df, test_full_df, val_full_df = get_splitted_data(data_path, train_ratio)

    # get columns needed for model training
    train_data = train_full_df[['Title', 'Encoded Labels']]
    val_data = val_full_df[['Title', 'Encoded Labels']]


    model_name = training_params['model_name']

    #get tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    combined_df = pd.concat([train_full_df, test_full_df, val_full_df] )
    
    _, classes, _ = multi_label_binarization(combined_df, 'Combined Labels')

    class2id, id2class = create_class_mappings(classes)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    num_classes = len(classes)
    model = AutoModelForSequenceClassification.from_pretrained(

    model_name, num_labels=num_classes,
            id2label=id2class, label2id=class2id,
                        problem_type = "multi_label_classification"
                        )
    

    tokenized_train_data = tokenization_processings(train_data, tokenizer)
    tokenized_val_data = tokenization_processings(val_data, tokenizer)

    # custom weighted loss
    class_weights = compute_class_weights(train_full_df)
    
    weighted_loss = WeightedLoss(class_weights)

    training_args_json= get_json_file(training_params_path)
    training_args_json.pop('model_name')
    training_args_json.pop('train_ratio')

    #load arguments
    training_args = TrainingArguments(**training_args_json)

  
    trainer = CustomTrainer( # Use the custom trainer class
        model=model,
        args=training_args,
        train_dataset=tokenized_train_data,
        eval_dataset=tokenized_val_data,
        data_collator=data_collator,
        weighted_loss = weighted_loss,
        compute_metrics=compute_metrics,
        callbacks = [TensorBoardCallback]
    )

    output_dir = training_params['output_dir']
    trainer.train()
    trainer.save_model(output_dir)  # Saves the model, configuration, and optimizer state
    tokenizer.save_pretrained(output_dir)  



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a multi-label classification model.")
    parser.add_argument(
        "--data_parameters_path",
        type=str,
        default="data_parameters.json",
        help="Path to the data parameters JSON file."
    )
    parser.add_argument(
        "--training_params_path",
        type=str,
        default="model_training_args.json",
        help="Path to the training parameters JSON file."
    )

    args = parser.parse_args()

    train_model(data_parameters_path=args.data_parameters_path, training_params_path=args.training_params_path)
