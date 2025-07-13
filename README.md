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

## How to Run This Project (Google Colab)

This project is designed to be run easily in Google Colab, which requires no local setup.

### 1. Open in Google Colab

- If you have the `.ipynb` file, upload it to your Google Drive and open it with Colab.
- If you are viewing this repository on GitHub, you can often open the notebook directly in Colab by replacing `github.com` with `colab.research.google.com/github/` in the URL.

### 2. Prepare the Dataset

- The required Python libraries (`pandas`, `scikit-learn`, etc.) are pre-installed in Google Colab.
- You will need to download the dataset into your Colab environment. The easiest way is to use `kagglehub`. Run the following code cell at the beginning of your notebook:

```python
# Install kagglehub
!pip install kagglehub --quiet

# Import kaggle libraries
import kagglehub

# Download the dataset
path = kagglehub.dataset_download("mlg-ulb/creditcardfraud")

# Read the creditcard.csv into pandas dataframe
df = pd.read_csv(path + '/creditcard.csv')
```

### 3. Run the Notebook

Once the dataset (`creditcard.csv`) is available in your environment, you can run all the cells in the notebook sequentially to execute the project pipeline.

## Key Results

The final model achieved strong performance on the unseen test set:

* **Recall:** 0.86
* **Precision:** 0.91
* **F1-Score:** 0.88
* **PR-AUC:** 0.89

These results indicate that the model is highly effective at identifying fraudulent transactions while maintaining a low rate of false positives.
