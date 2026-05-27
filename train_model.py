import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

def train_spam_model():
    print("Loading dataset...")
    # Load data
    df = pd.read_csv("dataset.csv")
    
    X = df['text']
    y = df['label']
    
    # 1. Convert text data into numerical vectors
    vectorizer = CountVectorizer()
    X_vectorized = vectorizer.fit_transform(X)
    
    # 2. Split data (Even with a small sample, we train on what we have)
    X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, random_state=42)
    
    # 3. Initialize and train the Naive Bayes Model
    print("Training the Naive Bayes model...")
    model = MultinomialNB()
    model.fit(X_vectorized, y) # Fitting on total sample for small demo datasets
    
    # 4. Save the model and vectorizer to disk
    with open("spam_model.pkl", "wb") as model_file:
        pickle.dump(model, model_file)
        
    with open("vectorizer.pkl", "wb") as vec_file:
        pickle.dump(vectorizer, vec_file)
        
    print("Model and Vectorizer successfully trained and saved!")

if __name__ == "__main__":
    train_spam_model()