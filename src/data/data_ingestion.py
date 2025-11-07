import pandas as pd
import numpy as np
import os

def data_ingestion(url : str)->pd.DataFrame:
    df=pd.read_csv(url)
    return df

def save_data(path :str,df: pd.DataFrame)-> None:
    os.makedirs(os.path.dirname(path), exist_ok=True) 
    # Save DataFrame to CSV
    df.to_csv(path, index=False)
    print(f"✅ Data saved successfully at: {path}")

    
def main():
    df=data_ingestion(r"C:/Users/mdmuz/.cache/kagglehub/datasets/anishdevedward/loan-approval-dataset\versions/1/loan_approval.csv")
    save_data(r'data/raw/raw_data.csv',df)

if __name__ == "__main__":
    main()
    
