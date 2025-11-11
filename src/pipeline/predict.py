import torch
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import tqdm
import numpy as np

def make_predictions(model: torch.nn.Module, dataloader: DataLoader, logger, output_file: str = None) -> pd.DataFrame:
    """
    Make predictions using the trained model and provided DataLoader.
    Save predictions with corresponding stock codes into a CSV file.
    
    Args:
        model (torch.nn.Module): The trained model.
        dataloader (DataLoader): DataLoader containing the data to make predictions on.
        output_file (str): The path where the predictions will be saved as a CSV file.
    
    Returns:
        np.ndarray: The predictions as a numpy array.
    """
    model.eval()  # Set the model to evaluation mode
    predictions = []
    stock_codes = []

    with torch.no_grad():  # Disable gradient calculation for inference
        for codes, features, _ in dataloader:
            outputs = model(features)  # Model prediction
            predictions.append(pd.Series(outputs.numpy().squeeze()))  # Convert to numpy and then to Pandas Series
            stock_codes.append(pd.Series(codes.numpy().squeeze()))  # Convert to numpy and then to Pandas Series

    # Concatenate the predictions and stock codes
    predictions = pd.concat(predictions, ignore_index=True)
    stock_codes = pd.concat(stock_codes, ignore_index=True)

    # Create a DataFrame with stock codes and predictions
    result_df = pd.DataFrame({
        'stock_code': stock_codes,
        'prediction': predictions
    })
    print(result_df)
    return result_df