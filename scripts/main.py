import pandas as pd
from data_preprocessing import create_data_pipeline, save_pipeline, load_pipeline, split_data, encode_response_variable
from ml_functions import training_pipeline, prediction_pipeline, evaluation_matrices, save_model, load_model
from helper_functions import logging
from scipy import sparse

def main():
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Load data
    df = pd.read_csv('./pswd/cleanpasswordlist.csv', dtype={'password': str})
    logging.info('Data loaded successfully.')

    # Handle NaN values
    df['password'] = df['password'].fillna('')  # Replace NaN values with an empty string
    logging.info('NaN values in password column handled.')

    # Feature selection
    X = df['password']
    y = df['strength']

    # Encode response variable
    y_encoded = encode_response_variable(y)

    # Create and fit the data processing pipeline
    pipeline = create_data_pipeline()
    X_transformed = pipeline.fit_transform(X)
    logging.info('Data processing pipeline created and applied to data.')

    # Check if X_transformed is sparse
    if sparse.issparse(X_transformed):
        logging.info("X_transformed is a sparse matrix.")

    # Save the pipeline for later use
    save_pipeline(pipeline, './pswd/artifacts/data_processing_pipeline.pkl')
    logging.info('Data processing pipeline saved.')

    # Split the data for training and validation
    X_train, X_val, y_train, y_val = split_data(X_transformed, y_encoded)

    # Train the best model
    best_model = training_pipeline(X_train, y_train)

    # Save the trained model
    save_model(best_model, './pswd/artifacts/best_classifier.pkl')
    logging.info('Model saved successfully.')

    # Make predictions
    predictions = prediction_pipeline(X_val, best_model)

    # Evaluate the model
    conf_matrix, acc_score, class_report = evaluation_matrices(X_val, y_val, best_model)

    logging.info('Model training, prediction, and evaluation completed.')

if __name__ == "__main__":
    main()