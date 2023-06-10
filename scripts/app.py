import streamlit as st
import pandas as pd
import pickle
import requests
import os
from PIL import Image
import logging

# Configure logging to write to a file in the current directory
logging.basicConfig(
    filename=os.path.join('./pswd/', 'logs/logfile_UI.txt'),
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Function to load artifacts
def load_artifact(filename):
    try:
        with open(filename, 'rb') as file:
            return pickle.load(file)
    except FileNotFoundError as e:
        logging.error(f"Artifact file not found: {filename}")
        st.error(f"Artifact file not found: {filename}")
        raise

# Function to load the pipeline and model
def load_pipeline(path):
    with open(path, 'rb') as file:
        pipeline = pickle.load(file)
    return pipeline

def load_model(path):
    with open(path, 'rb') as file:
        model = pickle.load(file)
    return model

def load_label_encoder(path):
    with open(path, 'rb') as file:
        label_encoder = pickle.load(file)
    return label_encoder

# Absolute path to the artifacts directory
artifact_dir = './pswd/artifacts'

# Absolute path to the images directory
images_dir = './pswd/images'

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Adhoc Prediction", "Batch Prediction"])

# Layout: Image on the left, title on the right
col1, col2 = st.columns([1, 3])
with col1:
    image_path = os.path.join(images_dir, 'password-strength-image.jpg')
    if os.path.exists(image_path):
        image = Image.open(image_path)
        st.image(image, use_column_width=True)
    else:
        st.warning(f"Image file not found: {image_path}")

with col2:
    st.title("Password Strength Classification")

# Navigation logic
if page == "Home":
    st.write("Welcome to the Password Strength Classification App.")
    st.write("Use the sidebar to navigate to Adhoc or Batch Prediction.")

elif page == "Adhoc Prediction":
    st.header("Enter password details:")
    password = st.text_input("Password")

    if st.button('Predict Strength'):
        if password:
            try:
                # Load the pipeline, model, and label encoder
                pipeline = load_pipeline(os.path.join(artifact_dir, 'data_processing_pipeline.pkl'))
                model = load_model(os.path.join(artifact_dir, 'best_classifier.pkl'))
                label_encoder = load_label_encoder(os.path.join(artifact_dir, 'label_encoder.pkl'))

                # Transform the input
                X_transformed = pipeline.transform([password])

                # Predict
                prediction = model.predict(X_transformed)
                logging.info(f"Raw prediction: {prediction}")

                # Map the prediction to human-readable labels
                label_map = {0: 'Weak', 1: 'Medium', 2: 'Strong'}
                predicted_strength = label_map.get(prediction[0], 'Unknown')

                st.subheader('Predicted Password Strength:')
                st.write(predicted_strength)
                logging.info(f"Prediction result displayed: {predicted_strength}")

                # Display the raw prediction
                st.write(f"Raw prediction: {prediction}")

            except Exception as e:
                st.error(f"Error occurred during prediction: {e}")
                logging.error(f"Error during prediction: {e}")
        else:
            st.warning("Please enter a password to classify.")

elif page == "Batch Prediction":
    st.header("Batch Prediction")
    uploaded_file = st.file_uploader("Upload your CSV file for batch prediction", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        logging.info(f"Batch file uploaded with {len(df)} records")

        if 'password' not in df.columns:
            st.error("CSV file must contain a 'password' column.")
            logging.error("Uploaded file does not contain 'password' column.")
        else:
            try:
                # Update this line with your FastAPI service URL
                response = requests.post("http://fastapi-container:8001/batch_predict", json={"data": df.to_dict(orient="list")})
                
                response.raise_for_status()
                predictions = response.json()
                output_df = pd.DataFrame(predictions)
                output_folder = os.path.join('./pswd/', 'Data', 'output')
                os.makedirs(output_folder, exist_ok=True)
                output_file_path = os.path.join(output_folder, 'batch_predictions.csv')
                output_df.to_csv(output_file_path, index=False)
                st.success(f"Batch predictions saved to {output_file_path}")
                logging.info(f"Batch predictions saved to {output_file_path}")
            except requests.exceptions.HTTPError as http_err:
                st.error(f"HTTP error occurred during batch prediction: {http_err}")
                logging.error(f"Batch prediction failed due to HTTP error: {http_err}")
            except requests.exceptions.RequestException as req_err:
                st.error("Error during batch prediction. Please check the API service.")
                logging.error(f"Batch prediction failed: {req_err}")
            except Exception as e:
                st.error(f"Error occurred during batch prediction: {e}")
                logging.error(f"Batch prediction failed: {e}")
