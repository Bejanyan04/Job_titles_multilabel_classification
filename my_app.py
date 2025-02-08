from fastapi import FastAPI
import torch

app = FastAPI()

# Load the trained model
model = torch.load("best_model")

@app.post("/predict")
async def predict(input_data: list):
"""
    classes = load_mlb_mapping()
  test_data = get_test_data()
  tokenizer = get_tokenizer()
  model = get_model() #get best model( finetuned huggingface model)
  
    result = model(torch.tensor(input_data)).tolist()
    return {"prediction": result}

"""
pass

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)
