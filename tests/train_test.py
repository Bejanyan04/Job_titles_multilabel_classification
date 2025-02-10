import pytest
import json
@pytest.fixture(scope="session")
def train_params():
    #json_path = os.getenv('TRAIN_PARAMS_PATH') 
    json_path = 'model_training_args.json'
    with open(json_path, "r") as f:
        return json.load(f)
      
def test_train_params_keys(train_params):
    required_keys = {'model_name', 'train_ratio', 'output_dir', 'learning_rate', 
                     'per_device_train_batch_size', 'per_device_eval_batch_size',
                    'num_train_epochs', 'weight_decay', 'eval_strategy', 'save_strategy',
                    'load_best_model_at_end', 'metric_for_best_model', 'logging_dir', 
                    'logging_steps', 'eval_steps', 'save_steps', 'report_to'}
    assert required_keys.issubset(train_params.keys()), "Missing required keys"


