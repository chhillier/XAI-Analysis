# Project Notes: Explaining a Confused EEG Model

## My Goal
*(Describe the main objective of the project here. What question are you trying to answer?)*

---

## The Problem I'm Simulating
*(Explain the dataset and the specific challenge you created, like the "confuser" class.)*

- **The Main Problem:**
- **My Hypothesis:**

---

## My Workflow
Here's the step-by-step process to get from nothing to an answer.

### 1. Make the Data
- **What:** Run the `generate_data.py` script in `Data-Generation`.
- **Output:** A big CSV file with all the simulated signals.

### 2. Prep the Data for the CNN
- **What:** Run the `convert_to_cnn_format.py` script.
- **Why:** This takes the CSV and turns it into clean `.npy` files for PyTorch.

### 3. Find the Best Model & Train It
- **What:** Run the `train_cv_cnn.py` script in `NeuralNets`.
- **Why:** This uses Optuna to find the best CNN architecture and saves the test set files.

### 4. Figure Out What the Model Was Thinking (XAI)
- **What:** Run the `xai_analysis.py` script in the `XAI` repo.
- **Why:** This is where the analysis happens. It loads the trained model and uses Captum to generate explanation plots.

---

## What I've Found So Far
*(Document your findings as you analyze the model's performance and the XAI plots.)*

- **The Results:**
- **The XAI Plot:**
- **My Takeaway:**

**Next Step:**

---

## How to Run This Again

1.  **Generate Data:**
    ```bash
    cd Data-Generation
    python generate_data.py
    python convert_to_cnn_format.py
    ```

2.  **Train Model:**
    ```bash
    cd ../NeuralNets
    python train_cv_cnn.py
    ```
    *(This creates `X_test.npy` and `y_test.npy`)*

3.  **Prep for XAI:**
    - `cd ../XAI`
    - Make a `test_data` folder.
    - Copy `../NeuralNets/X_test.npy` and `../NeuralNets/y_test.npy` into `test_data/`.
    - Run `python sync_model.py` to copy the trained model over.

4.  **Run XAI:**
    ```bash
    python xai_analysis.py
    ```