# Student Loan Risk with Deep Learning

This project analyzes student loan risk using a deep learning approach. It demonstrates the preparation of data, training of a neural network, and evaluation of model performance to classify loan risks.

## Table of Contents
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Dependencies](#dependencies)
- [Project Workflow](#project-workflow)
- [How to Run](#how-to-run)
- [Results](#results)
- [Acknowledgments](#acknowledgments)

## Introduction
The goal of this project is to develop a machine learning model that predicts student loan risks using features from a dataset. A deep learning approach is implemented with TensorFlow/Keras to explore the effectiveness of neural networks in risk classification tasks.

## Dataset
The project uses the `student-loans.csv` dataset. It contains information about student loans, which is used to define features and target variables for training the model.

### Features:
- Loan-related metrics (e.g., income, debt-to-income ratio, etc.)

### Target:
- Binary classification of loan risk (e.g., `low risk` vs. `high risk`).

## Dependencies
Ensure you have the following Python packages installed:

- `pandas`
- `tensorflow`
- `scikit-learn`
- `pathlib`

You can install the required packages using the command:

```bash
pip install pandas tensorflow scikit-learn
```

## Project Workflow
1. **Data Preparation:**
   - Load the `student-loans.csv` dataset into a Pandas DataFrame.
   - Perform exploratory data analysis (EDA) to identify features and target variables.
   - Standardize the data for better model performance.

2. **Model Development:**
   - Build a Sequential neural network model using TensorFlow/Keras.
   - Define the input, hidden, and output layers.

3. **Training:**
   - Split the data into training and testing sets.
   - Train the model using the training data.

4. **Evaluation:**
   - Evaluate the model's performance on the test data using metrics like classification report.

## How to Run
1. Clone the repository or download the notebook file.
2. Open the `student_loans_with_deep_learning.ipynb` notebook in Jupyter Notebook or JupyterLab.
3. Run each cell sequentially to execute the code and follow the workflow.
4. The model's performance and results will be displayed in the output cells.

## Results
The trained model outputs metrics such as precision, recall, and F1-score for loan risk classification. The results help in understanding the model's effectiveness and areas for improvement.

## Acknowledgments
This project utilizes data and resources from [BC-EDX](https://static.bc-edx.com). The deep learning model is built using TensorFlow and the project follows a systematic approach to machine learning.
