# Job Titles Multi-Label Classification

## Project Overview
This project tackles the task of **multi-label classification** for job titles using a dataset consisting of Job Titles and corresponding Labels. Each job title in the dataset may have one or more labels corresponding to organizational roles.

---

## Data Analysis and Preprocessing

### Key Steps:
1. **Class Imbalance**:
   - Minority classes were identified, which contributed to poor metrics for these labels.
   - **Weighted loss techniques** were applied during training to mitigate this issue, but further improvements are required.

2. **Handling Missing Labels**:
   - Rows with fully missing labels were removed due to their minimal occurrence and negligible impact on training.

3. **Stratified Splitting**:
   - The dataset was split into **training (80%)**, **validation (10%)**, and **test (10%)** sets using a **stratified approach** to preserve label distributions across splits.

---

## Model Selection

The **`microsoft/deberta-v3-small`** model was selected for this task because of the following reasons:

- **Lightweight Architecture**:
  - The model has approximately **22M parameters**, making it efficient for training and inference.
  
- **Pretrained Capability**:
  - Pretrained on a large corpus, the model generalizes well to domain-specific tasks like job title classification.

- **Disentangled Attention Mechanism**:
  - This allows the model to capture relationships between tokens effectively, which is critical for understanding job titles.

---

## Training Details

### Configuration:
- **Hardware**: A T4 GPU with 16GB memory was used; the model utilized ~7GB during training.
- **Fine-Tuning**:
  - Full fine-tuning was performed for **10 epochs**.
  - A custom Trainer was implemented using **weighted binary cross-entropy loss** to handle class imbalance.
- **Early Stopping**:
  - Early stopping was applied based on the **F1 score**, and the best model was saved.

### Performance:
- **Training Time**:
  - Each epoch took approximately **15 minutes**.
- **Monitoring**:
  - Training metrics (loss and F1 score) were logged using **TensorBoard**.

---

## Inference

- The model's inference was tested on the full dataset and computed metrics for each label independently.
- Inference is lightweight and can be efficiently performed on a **CPU**.

---

## Results and Future Improvements

### Current Results:
- The model performed well overall but exhibited poor metrics for **minority classes** despite the use of weighted loss techniques.

### Future Improvements:
1. Experiment with **advanced class weighting techniques** or **data augmentation** for minority classes.
2. Evaluate **alternative transformer architectures** or **ensemble methods** for further performance gains.
3. Compare the transformer-based model with **classical machine learning algorithms** to assess trade-offs in complexity and performance.


## Model Deployment and Inference from Hugging Face Hub

After training, the **best model** was uploaded to the **Hugging Face Hub** for easy access and inference. The model and tokenizer were pushed to the repository under the name **`ann0401/Job_Titles_Classification`**, making it publicly available for use.

#### Steps for Inference:
To use the trained model for inference, the **model** and **tokenizer** can be loaded directly from the Hugging Face Hub:

