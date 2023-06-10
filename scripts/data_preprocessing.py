import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import pickle
import pandas as pd
import numpy as np

def create_data_pipeline():
    """
    Creates a data processing pipeline for password data.

    This pipeline includes TF-IDF vectorization.

    Returns:
        Pipeline: The created data processing pipeline.
    """
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(analyzer='char', ngram_range=(3, 5)))  # TF-IDF on character level with n-grams
    ])
    return pipeline

def save_pipeline(pipeline, filename):
    """
    Saves the machine learning pipeline to a file.

    Args:
      pipeline (object): The machine learning pipeline to save.
      filename (str): The name of the file to save the pipeline to.
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    with open(filename, 'wb') as f:
        pickle.dump(pipeline, f)
    
    print(f'Pipeline saved to: {filename}')

def load_pipeline(filename):
    """
    Loads a machine learning pipeline from a file.

    Args:
      filename (str): The name of the file containing the pipeline.

    Returns:
      object: The loaded machine learning pipeline.
    """
    with open(filename, 'rb') as f:
        pipeline = pickle.load(f)
    print(f'Pipeline loaded from: {filename}')
    return pipeline

def encode_response_variable(y):
    """
    Encodes the response variable (y) using label encoding.

    Args:
        y (pd.Series or np.ndarray): The response variable data.

    Returns:
        np.ndarray: The encoded response variable.
    """
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Save the label encoder for later use
    with open('./pswd/artifacts/label_encoder.pkl', 'wb') as f:
        pickle.dump(label_encoder, f)

    print('Labels encoded for the response variable.')
    return y_encoded

from scipy import sparse

def split_data(X, y, test_size=0.2, random_state=42):
    """
    Splits the data into training and testing sets.

    Args:
      X (pd.Series, np.ndarray, or sparse matrix): The features data.
      y (pd.Series or np.ndarray): The target labels.
      test_size (float, optional): Proportion of data for the testing set. Defaults to 0.2.
      random_state (int, optional): Seed for random splitting. Defaults to 42.

    Returns:
      tuple: A tuple containing the training and testing data splits (X_train, X_test, y_train, y_test).
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    print('Data is split into training and testing sets.')
    return X_train, X_test, y_train, y_test
