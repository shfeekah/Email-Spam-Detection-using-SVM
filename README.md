# 📧 Email Spam Detection using Linear SVM

## Overview

This project is a Machine Learning-based Email Spam Detection system developed in Python using a **Linear Support Vector Machine (Linear SVM)** classifier. It classifies incoming text messages as either **Spam** or **Ham (Not Spam)** by transforming text into numerical features using **TF-IDF (Term Frequency–Inverse Document Frequency)**.

The project demonstrates the complete machine learning workflow, including data preprocessing, feature extraction, model training, evaluation, model persistence, and prediction on new user-provided messages.

---

## Features

* Load and preprocess the SMS Spam Collection dataset
* Clean and prepare text data
* Convert text into numerical features using TF-IDF
* Train a Linear SVM classifier
* Evaluate model performance using:

  * Accuracy
  * Precision
  * Recall
  * F1-Score
  * Confusion Matrix
* Save the trained model and vectorizer
* Predict whether a custom message is **Spam** or **Ham**

---

## Technologies Used

* Python 3
* Pandas
* Scikit-learn
* Joblib / Pickle
* VS Code

---

## Project Structure

```text
Email Spam Detection/
│
├── dataset/
│   └── spam.csv
│
├── model/
│   ├── spam_model.pkl
│   └── tfidf_vectorizer.pkl
│
├── train.py
├── predict.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Dataset

This project uses the **SMS Spam Collection Dataset**, which contains labeled SMS messages categorized as:

* **Ham** – Legitimate messages
* **Spam** – Unwanted or promotional messages

---

## Machine Learning Workflow

1. Load the dataset
2. Clean and preprocess the data
3. Convert text to TF-IDF features
4. Split the dataset into training and testing sets
5. Train a Linear SVM classifier
6. Evaluate model performance
7. Save the trained model
8. Predict new messages

---

## Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/Email-Spam-Detection-using-SVM.git
```

Move into the project folder:

```bash
cd Email-Spam-Detection-using-SVM
```

Install the required packages:

```bash
pip install -r requirements.txt
```

---

## Train the Model

Run:

```bash
python train.py
```

This will:

* Train the Linear SVM model
* Evaluate the model
* Save:

  * `spam_model.pkl`
  * `tfidf_vectorizer.pkl`

---

## Predict a New Message

Run:

```bash
python predict.py
```

Example:

```text
Enter your message:
Congratulations! You have won a free iPhone.
```

Output:

```text
 Prediction: SPAM -_-
```

Another example:

```text
Enter your message:
Hey, are we still meeting tomorrow at 5 PM?
```

Output:

```text
✅ Prediction: HAM <3
```

---

## Model Performance

| Metric           | Result     |
| ---------------- | ---------- |
| Accuracy         | **98.65%** |
| Precision (Spam) | **0.99**   |
| Recall (Spam)    | **0.91**   |
| F1-Score (Spam)  | **0.95**   |

### Confusion Matrix

```text
[[965   1]
 [ 14 135]]
```

The model correctly classified **1100 out of 1115** test messages, achieving **98.65% accuracy**.

---

## Future Improvements

* Train on a larger email dataset instead of SMS messages
* Experiment with other classifiers such as Naive Bayes, Logistic Regression, and Random Forest
* Perform hyperparameter tuning
* Build a graphical user interface (GUI)
* Develop a web application using Flask or FastAPI
* Deploy the model to the cloud

---

## Learning Outcomes

Through this project, I learned how to:

* Build an end-to-end machine learning pipeline
* Preprocess text data
* Apply TF-IDF feature extraction
* Train and evaluate a Linear SVM classifier
* Save and reuse trained machine learning models
* Develop an interactive prediction application in Python

---

## Author

**Shafika**

Machine Learning & Artificial Intelligence Enthusiast

Information and Communication Engineering (ICE)

Bangladesh University of Professionals (BUP)
