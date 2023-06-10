import pickle
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import pandas as pd
import numpy as np

def training_pipeline(X_train, y_train):
    """
    Train a model using XGBoost.

    Args:
        X_train (sparse matrix or np.ndarray): The training features.
        y_train (np.ndarray): The training labels.
    
    Returns:
        XGBClassifier: The trained XGBoost model.
    """
    model = XGBClassifier(
        use_label_encoder=False,
        eval_metric='mlogloss',
        tree_method='hist',      # Use CPU-based method
        device='cpu'
    )
    print(f"{X_train.shape}, {y_train.shape}")
    model.fit(X_train, y_train)
    print("Model trained successfully.")
    return model

def save_model(model, path):
    """
    Save the trained model to the specified path.

    Args:
        model: The trained model to be saved
        path (str): Path to save the model file
    """
    with open(path, 'wb') as file:
        pickle.dump(model, file)
    print(f"Model saved to: {path}")

def load_model(path):
    """
    Loads a pickled model from the specified path.

    Args:
        path (str): Path to the pickled model file.

    Returns:
        object: The loaded model object.
    """
    with open(path, 'rb') as file:
        model = pickle.load(file)
    print(f'Model loaded from: {path}')
    return model

def prediction_pipeline(X_val, model):
    """
    Makes predictions on the data using the provided model.

    Args:
        X_val (pd.DataFrame): Validation data features.
        model (XGBClassifier): The trained model.

    Returns:
        np.ndarray: Array of predicted target labels.
    """
    predictions = model.predict(X_val)
    return predictions

def evaluation_matrices(X_val, y_val, model):
    """
    Calculates and logs evaluation metrics for the model.

    Args:
        X_val (pd.DataFrame): Validation data features.
        y_val (pd.Series or np.ndarray): Validation data target labels.
        model (XGBClassifier): The trained model.

    Returns:
        tuple: A tuple containing the confusion matrix, accuracy score, and classification report.
    """
    pred_vals = model.predict(X_val)

    # Calculate the confusion matrix, accuracy score, and classification report
    conf_matrix = confusion_matrix(y_val, pred_vals)
    acc_score = accuracy_score(y_val, pred_vals)
    class_report = classification_report(y_val, pred_vals)

    # Print evaluation metrics
    print("Confusion Matrix:\n", pd.DataFrame(conf_matrix, index=np.unique(y_val), columns=np.unique(y_val)))
    print("Accuracy Score: ", acc_score)
    print("Classification Report:\n", class_report)

    return conf_matrix, acc_score, class_report