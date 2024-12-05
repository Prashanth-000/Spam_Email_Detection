# model.py
import pandas as pd
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pickle

# Download necessary NLTK data
nltk.download('stopwords')
nltk.download('wordnet')

# Sample dataset (same data as before, replace with your actual data)
emails = [
    ("Free money!!!", 1),
    ("Hi, how are you?", 0),
    ("Win a lottery now", 1),
    ("Meeting at 10 AM", 0),
    ("Earn $1000 per day", 1),
    ("Lunch at 1 PM?", 0),
    ("Cheap meds online", 1),
    ("Your invoice", 0),
    ("Limited time offer!!!", 1),
    ("Project deadline tomorrow", 0),
    ("Get rich quick!!!", 1),
    ("Family reunion next week", 0),
    ("Exclusive discount for you", 1),
    ("Please review the attached report", 0),
    ("You won a free vacation!", 1),
    ("Office party this Friday", 0),
    ("Lowest price guarantee", 1),
    ("Team meeting rescheduled", 0),
    ("Claim your free gift now", 1),
    ("Urgent: Your account needs attention", 1),
    ("Can we reschedule our appointment?", 0),
    ("Get out of debt fast!", 1),
    ("New product launch event", 0),
    ("Congratulations, you have been selected!", 1),
    ("Important notice regarding your policy", 0),
    ("You are pre-approved for a loan", 1),
    ("Dinner plans for tonight?", 0),
    ("Win big with our casino games", 1),
    ("Your subscription is expiring soon", 0)
]

# Convert to DataFrame
data = pd.DataFrame(emails, columns=['Email', 'Spam'])

# Text preprocessing function
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.strip()
    tokens = text.split()
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stopwords.words('english')]
    return ' '.join(tokens)

# Apply preprocessing
data['Processed_Email'] = data['Email'].apply(preprocess_text)

# Vectorizing the text using TF-IDF
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(data['Processed_Email'])
y = data['Spam']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model (Logistic Regression)
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluate the model
from sklearn.metrics import accuracy_score, classification_report
y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(classification_report(y_test, y_pred))

# Save the trained model and vectorizer to disk
with open('spam_classifier_model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)

with open('vectorizer.pkl', 'wb') as vectorizer_file:
    pickle.dump(vectorizer, vectorizer_file)

print("Model and vectorizer saved successfully.")
