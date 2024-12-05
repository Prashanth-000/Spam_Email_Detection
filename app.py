# app.py
from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load the saved model and vectorizer
model = pickle.load(open('spam_classifier_model.pkl', 'rb'))  # Load the trained model
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))  # Load the trained vectorizer

@app.route('/', methods=['GET', 'POST'])
def index():
    # This route will display the email input form
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the email content from the form
    email_content = request.form['emailInput']
    
    # Vectorize the email content (transform it using the vectorizer)
    email_vectorized = vectorizer.transform([email_content])
    
    # Predict whether the email is spam (1) or not spam (0)
    prediction = model.predict(email_vectorized)
    
    # Display result: "Spam" or "Not Spam"
    result = "Spam" if prediction == 1 else "Not Spam"
    
    # Render result page with the prediction
    return render_template('result.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
