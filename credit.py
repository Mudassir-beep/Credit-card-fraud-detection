import pandas as pd
import numpy as np
import streamlit as st
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load data and train the model (ideally should be pre-trained & loaded in production)
df = pd.read_csv('C:/Users/Mudassir/Downloads/creditcard.csv')
y = df['Class']
x = df.drop(['Class'], axis=1)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

lr = LogisticRegression(max_iter=5000)
lr.fit(x_train, y_train)

# Evaluate model
y_pred = lr.predict(x_test)
acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)

# Streamlit UI
st.title("Credit Card Fraud Detection")

# Image
try:
    img = Image.open('C:/Users/Mudassir/Downloads/Credit_card.png')
    st.image(img, width=200)
except Exception as e:
    st.warning("⚠️ Could not load image. Check file path or format.")

# Input prompt
st.write("Enter credit card transaction data (29 values):")
st.markdown("Separate values using **spaces**, **commas**, or both. Example: `0.1 -1.2 0.5 ...`")

user_input = st.text_input('Enter the input values:')

# Updated input cleaner
def clean_input(text):
    try:
        # Normalize characters
        text = text.replace('–', '-').replace('−', '-')
        text = text.replace('\t', ' ').replace(',', ' ')
        
        # Split and debug
        parts = text.split()
        st.write(f"Parsed values ({len(parts)}):", parts)  # 👈 show in Streamlit

        # Convert to float
        return [float(p) for p in parts]
    
    except ValueError as ve:
        st.error(f"⚠️ ValueError: Could not convert part of the input to float. Problem: {ve}")
        raise
    except Exception as e:
        st.error(f"⚠️ Unexpected error during input processing: {e}")
        raise



# Prediction block
if st.button('Predict'):
    try:
        input_values = clean_input(user_input)

        if len(input_values) != 29:
            st.error(f"⚠️ Expected 29 values, but got {len(input_values)}.")
        else:
            input_array = np.asarray(input_values).reshape(1, -1)
            prediction = lr.predict(input_array)

            if prediction[0] == 0:
                st.success("✅ Prediction: Legitimate Transaction")
            else:
                st.error("🚨 Prediction: Fraudulent Transaction")
    except ValueError:
        st.error("❌ Please ensure all inputs are numeric and correctly formatted.")
    except Exception as e:
        st.error(f"Unexpected error: {e}")
