import joblib

# Load  trained model
model = joblib.load("model/spam_model.pkl")

# Load TF-IDF 
vectorizer = joblib.load("model/tfidf_vectorizer.pkl")

#  enter a msg
message = input("Enter the email content : ")

# Convert msg into numbers
message_vector = vectorizer.transform([message])

# Predict
prediction = model.predict(message_vector)

# Display result
if prediction[0] == 1:
    print("\n Prediction: SPAM -_-")
else:
    print("\n Prediction: HAM <3 ")