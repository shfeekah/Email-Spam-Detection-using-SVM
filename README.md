# 📧 Email Spam Detection using Linear SVM

A Machine Learning project that classifies SMS messages as **Spam** or **Ham (Not Spam)** using a **Linear Support Vector Machine (Linear SVM)** classifier and **TF-IDF** feature extraction.

---

## 📖 Overview

Email and SMS spam messages are a common cybersecurity issue. This project demonstrates how Machine Learning can automatically detect spam messages by learning patterns from labeled text data.

The workflow includes data preprocessing, feature extraction, model training, evaluation, model saving, and prediction on new user-provided messages.

---

## ✨ Features

- 📥 Load and preprocess the SMS Spam Collection dataset
- 🧹 Clean and prepare text data
- 🔤 Convert text into numerical features using TF-IDF
- 🤖 Train a Linear Support Vector Machine (Linear SVM) classifier
- 📊 Evaluate model performance using:
  - Accuracy
  - Precision
  - Recall
  - F1-Score
  - Confusion Matrix
- 💾 Save the trained model and TF-IDF vectorizer
- ✉️ Predict whether a custom message is Spam or Ham

---

## 🛠 Technologies Used

- Python 3
- Pandas
- NumPy
- Scikit-learn
- Joblib
- VS Code

---

## 📂 Project Structure

```
Email-Spam-Detection/
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
└── README.md
```

---

## 📊 Dataset

This project uses the **SMS Spam Collection Dataset**, which contains labeled SMS messages categorized as:

- **Ham** – Legitimate messages
- **Spam** – Promotional or unwanted messages

---

## ⚙️ Machine Learning Workflow

1. Load the dataset
2. Clean and preprocess the text
3. Convert text into TF-IDF features
4. Split the dataset into training and testing sets
5. Train a Linear SVM classifier
6. Evaluate model performance
7. Save the trained model
8. Predict new messages

---

## 🚀 Installation

Clone the repository:

```bash
git clone https://github.com/SHFEEKAH/Email-Spam-Detection-using-SVM.git
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

## ▶️ Training the Model

Run:

```bash
python train.py
```

This will:

- Train the Linear SVM classifier
- Evaluate its performance
- Save:
  - `spam_model.pkl`
  - `tfidf_vectorizer.pkl`

---

## 🔍 Making Predictions

Run:

```bash
python predict.py
```

Example:

```
Enter a message:

Congratulations! You've won a free iPhone.

Prediction:
Spam
```

---

## 📈 Evaluation Metrics

The model is evaluated using:

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix

These metrics help measure how well the classifier distinguishes spam from legitimate messages.

---

## 🎯 Future Improvements

- Build a Flask or Streamlit web application
- Support email spam detection in addition to SMS
- Experiment with additional algorithms:
  - Naive Bayes
  - Logistic Regression
  - Random Forest
- Hyperparameter tuning
- Deploy the model online

---

## 👩‍💻 Author

**Shafika Binte Ismail**

Information and Communication Engineering (ICE)  
Bangladesh University of Professionals (BUP)

---

## ⭐ Acknowledgements

- Scikit-learn
- Pandas
- SMS Spam Collection Dataset
