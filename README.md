# 📧 Spam Email Detection using Logistic Regression

A lightweight Flask web app that detects whether an input email is Spam or Not Spam using Natural Language Processing (NLP) and a Logistic Regression model. Built with Python, scikit-learn, and Flask.

## 🗂️ Project Structure

```

FAKE_EMAIL_DET/
├── static/
│   └── styles.css                      - Styling for the web app
├── templates/
│   ├── index.html                      - Email input form page
│   └── result.html                     - Result display page
├── app.py                              - Flask application
├── model.py                            - ML model training script
├── spam_classifier_model.pkl           - Saved ML model
├── vectorizer.pkl                      - Saved TF-IDF vectorizer
├── Spam_email.py                       - (Optional) extra utilities
├── Testing_Spam.txt                    - Sample spam email
├── Testing_Not_Spam.txt                - Sample non-spam email
├── Screenshot 2024-12-05 150703.png    - Screenshot of input form
├── Screenshot 2024-12-05 150714.png    - Screenshot of result display
├── README.md                           - Project documentation
```
```

## 🚀 Features

- Logistic Regression classifier for email spam detection
- Text preprocessing: lowercase, punctuation/digit removal, lemmatization, stopword removal
- Vectorization using TF-IDF
- Clean UI using HTML/CSS and Flask
- Model and vectorizer saved using pickle

```
```
## 📷 Screenshots

Screenshots of the app interface:

Input Form:
![Input Form](./Screenshot%202024-12-05%20150703.png)


Prediction Result:
![Input Form](./Screenshot%202024-12-05%20150714.png)

```
## 🧠 How It Works

1. Preprocess email text: lowercase, strip digits/punctuation, remove stopwords, lemmatize
2. Vectorize using TfidfVectorizer
3. Train Logistic Regression model on labeled email samples
4. Save model and vectorizer
5. Web form accepts user input → transformed and classified → result shown as "Spam" or "Not Spam"

## 🔧 Setup Instructions

1. Make sure you have Python 3.7+ installed

2. Install dependencies:
   pip install flask scikit-learn pandas nltk

3. (Optional) Download NLTK stopwords and lemmatizer in `model.py`:
   nltk.download('stopwords')
   nltk.download('wordnet')

4. Run the app:
   python app.py

5. Open your browser and go to:
   http://127.0.0.1:5000

## 🧪 Testing

Use the provided files:
- Testing_Spam.txt (contains a sample spam email)
- Testing_Not_Spam.txt (contains a sample genuine email)

Copy content into the form and submit to see predictions.

## 🏗️ Retrain the Model

If you want to retrain from scratch or modify data, just run:
   python model.py

This will:
- Preprocess and vectorize the text
- Train the Logistic Regression model
- Save spam_classifier_model.pkl and vectorizer.pkl
```
```
## ✍️ Author

PRASHANTHA – Computer Science Engineer & Developer

## 📌 Future Enhancements

- Add prediction probability/confidence
- Use a large dataset like Enron email dataset
- Deploy to Heroku or Render
- Improve UI with result graphs or classification explanation

```

## 📜 License

This project is licensed under the MIT License.
