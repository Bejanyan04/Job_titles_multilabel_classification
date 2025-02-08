from fastapi import FastAPI
import torch

app = FastAPI()

# Load the trained model
model = torch.load("best_model")

@app.post("/predict")
async def predict(input_data: list):
    result = model(torch.tensor(input_data)).tolist()
    return {"prediction": result}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)
