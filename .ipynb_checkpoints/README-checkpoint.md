# Credit Card Fraud Detection for Cambodian E-Commerce

This project implements a complete machine learning pipeline to detect fraudulent credit card transactions, with a special focus on securing the growing e-commerce landscape in Cambodia. The model is built using a real-world dataset and is designed to handle the severe class imbalance inherent in fraud detection tasks.

## Project Overview

The primary goal is to develop a robust supervised classification model that can accurately distinguish between legitimate and fraudulent transactions. This helps protect both consumers and businesses, fostering trust in the digital economy.

The project follows these key steps:

1. **ETL (Extract, Transform, Load):** Data is loaded, cleaned, and preprocessed.
2. **Feature Engineering:** The class imbalance is addressed using the Synthetic Minority Over-sampling Technique (SMOTE).
3. **Modeling:** A Random Forest Classifier is trained on the prepared data.
4. **Hyperparameter Tuning:** `RandomizedSearchCV` is used to find the optimal model configuration.
5. **Evaluation:** The model's performance is assessed using metrics suitable for imbalanced data, such as Precision, Recall, and the Precision-Recall Area Under Curve (PR-AUC).

## Dataset

This project uses the "Credit Card Fraud Detection" dataset from Kaggle.

* **Source:** [Kaggle Dataset Link](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
* **Content:** The dataset contains 284,807 transactions, of which only 492 (0.172%) are fraudulent. For privacy, most features are anonymized PCA components (`V1` to `V28`).

## How to Run This Project on a Virtual Server

This guide explains how to set up and run the project on a remote Linux server.

### 1. Initial Server Setup

First, connect to your server via SSH. Then, install the necessary tools.

```bash
# Update package lists
sudo apt update && sudo apt upgrade -y

# Install Python, pip, venv, and Git
sudo apt install python3-pip python3-venv git jupyterlab -y
```

### 2. Clone the Project Repository

Clone this repository to your server. Navigate into the **Machine-Learning-for-Credit-Card-Fraud-Detection** directory!

```bash
git clone https://github.com/jemmy-72/Machine-Learning-for-Credit-Card-Fraud-Detection
cd Machine-Learning-for-Credit-Card-Fraud-Detection
```

### 3. Set Up the Python Environment

Create a virtual environment to manage the project's dependencies.

```bash
# Create a virtual environment
python3 -m venv venv

# Activate the environment
source venv/bin/activate

# Install the required libraries
pip install -r requirements.txt
```

### 4. Run Jupyter Lab in a Session

To ensure your notebook keeps running even if you disconnect, use `tmux`.

```bash
# Install tmux
sudo apt install tmux -y

# Start a new tmux session
tmux new -s jupyter

# Inside the tmux session, start Jupyter Lab
# (Ensure your venv is still active)
jupyter lab --ip=0.0.0.0 --port=8888 --no-browser
```

### 5. Access and Run the Notebook

- In your local web browser, navigate to `http://yourip:8888`.
- When prompted, enter the token provided in your terminal output.
- You can also enter the token directly on the url, for example: `http://yourip:8888/lab?token=37a87ed0da1858c0942181220bc9c2c68bf634ce70ac17fc`
- Open the `Final_Project.ipynb` file and run the cells.

**Note on Runtime:** The hyperparameter tuning cell (`Step 4` in the notebook) is computationally intensive. Please expect it to take **60-120 minutes** to complete on a standard multi-core CPU. The specification that I use was CPU-Optimized / 8 GB / 4 vCPUs, 50 GB Disk / SGP1 - Ubuntu 24.10 x64. The time may varies depending on the specifications.

To safely detach from the `tmux` session, press **Ctrl+b**, then **d**.

## Key Results

The final model achieved strong performance on the unseen test set:

* **Recall:** 0.84
* **Precision:** 0.86
* **F1-Score:** 0.85
* **PR-AUC:** 0.87

These results indicate that the model is highly effective at identifying fraudulent transactions while maintaining a low rate of false positives.