import streamlit as st
import pickle

# Set up page configuration
st.set_page_config(page_title="SMS Spam Detector", page_icon="✉️", layout="centered")

# Load the trained model and vectorizer
@st.cache_resource
def load_assets():
    try:
        with open("spam_model.pkl", "rb") as model_file:
            model = pickle.load(model_file)
        with open("vectorizer.pkl", "rb") as vec_file:
            vectorizer = pickle.load(vec_file)
        return model, vectorizer
    except FileNotFoundError:
        return None, None

model, vectorizer = load_assets()

# UI Layout
st.title("✉️ SMS Spam Detection System")
st.write("Enter an SMS message below to check if it's safe (Ham) or fraudulent (Spam).")

st.markdown("---")

# User Input text area
user_input = st.text_area("Type your SMS message here:", height=150, placeholder="e.g., Congratulation! You won a free lottery...")

if st.button("Analyze Message"):
    if not model or not vectorizer:
        st.error("⚠️ Error: Model files not found! Please run 'python train_model.py' first in your terminal.")
    elif user_input.strip() == "":
        st.warning("Please enter some text to analyze.")
    else:
        # 1. Preprocess and Vectorize the user input
        data = [user_input]
        vectorized_input = vectorizer.transform(data)
        
        # 2. Predict using the model
        prediction = model.predict(vectorized_input)[0]
        # Get probability scores
        probabilities = model.predict_proba(vectorized_input)[0]
        spam_prob = probabilities[1] if len(probabilities) > 1 else 0.0
        
        st.markdown("### **Prediction Result:**")
        
        # 3. Output back to the UI
        if prediction == "spam":
            st.error(f"🚨 **SPAM DETECTED!** This message looks suspicious.")
        else:
            st.success(f"✅ **HAM (SAFE):** This message looks completely normal.")

st.markdown("---")
st.caption("Built with Python, Scikit-Learn, and Streamlit.")