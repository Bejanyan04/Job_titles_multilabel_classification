import pytest
import json
from transformers import AutoTokenizer, AutoModelForSequenceClassification



@pytest.fixture(scope="session")
def inference_params():
    #json_path =  os.getenv('INFERENCE_PARAMS_PATH')
    json_path ='inference_params.json'
    with open(json_path, "r") as f:
        return json.load(f)

def test_inference_params_keys(inference_params):
    required_keys = {"data_path", "trained_model_hub_repo_id", "train_ratio","trained_model_path"}
    assert required_keys.issubset(inference_params.keys()), "Missing required keys"

@pytest.fixture
def load_model(inference_params):
    
    """Fixture to load the model and tokenizer."""
    #model_path = inference_params.get("trained_model_path")
    model_path =  inference_params.get("trained_model_hub_repo_id")

    try:
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        return model, tokenizer
    except Exception as e:
        pytest.fail(f"Model loading failed: {e}")

def test_model_not_empty(load_model):
    """Test if the model loads correctly and is not empty."""
    model, _ = load_model
    num_params = sum(p.numel() for p in model.parameters())
    assert num_params > 0, "Model has no parameters! It might be corrupted."

def test_model_behaviour(load_model):
    model, tokenizer = load_model

    # Test tokenizer functionality
    sample_text = "Engineer"
    tokenized_input = tokenizer(sample_text, return_tensors="pt")
    assert "input_ids" in tokenized_input, "Tokenizer should return input_ids"
    assert "attention_mask" in tokenized_input, "Tokenizer should return attention_mask"

    # Test model functionality
    outputs = model(**tokenized_input)
    assert hasattr(outputs, "logits"), "Model should return logits"
    assert outputs.logits.shape[0] == 1, "Model output should have batch size 1"