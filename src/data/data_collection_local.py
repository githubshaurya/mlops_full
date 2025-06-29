import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
import yaml


def load_params(filepath : str) -> float:
    try:
        with open(filepath,"r") as file:
            params = yaml.safe_load(file)
        return params["data_collection"]["test_size"]
    except Exception as e:
        raise Exception(f"Error loading parameters from {filepath}:{e}")

def load_data(filepath : str) -> pd.DataFrame :
    try:
        return pd.read_csv(filepath)
    except Exception as e:
        raise Exception(f"Error loading data from {filepath} :{e}")

def split_data(data : pd.DataFrame, test_size: float) -> tuple[pd.DataFrame,pd.DataFrame]:
    try:
        return train_test_split(data, test_size= test_size, random_state=42)
    except Exception as e:
        raise Exception(f"Error splittin data : {e}")

def save_data(df : pd.DataFrame, filepath: str) -> None:
    try:
        df.to_csv(filepath,index=False)
    except Exception as e:
        raise Exception(f"Error saving data to {filepath} :{e}")
    

def main():
    # Use local file instead of downloading from GitHub
    data_filepath = "water_potability.csv"  # Local file in project root
    params_filepath = "params.yaml"
    raw_data_path = os.path.join("data","raw")
    
    try:
        data = load_data(data_filepath)
        test_size = load_params(params_filepath)
        train_data,test_data = split_data(data, test_size)

        os.makedirs(raw_data_path, exist_ok=True)

        save_data(train_data,os.path.join(raw_data_path,"train.csv"))
        save_data(test_data, os.path.join(raw_data_path,"test.csv"))
        
        print(f"Data split successfully!")
        print(f"Train data shape: {train_data.shape}")
        print(f"Test data shape: {test_data.shape}")
        
    except Exception as e:
        raise Exception(f"An error occurred :{e}")
    
if __name__ == "__main__":
    main() 