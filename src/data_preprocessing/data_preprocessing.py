import  numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
import yaml


def reading_yml(path :str )-> float:
        test_size = yaml.safe_load(open(path,'r'))['data_preprocessing']['test_size']
        return test_size

def load_data(url : str )-> pd.DataFrame:
    df=pd.read_csv(url)
  
    return df

def data_preprocessing(df :pd.DataFrame) ->pd.DataFrame:
    df.drop(columns=["name"],inplace=True)
    df['loan_approved']=df['loan_approved'].map({False:0,True:1})
    return df

def save_data(df :pd.DataFrame,train_path :str,test_path :str,test_size:float)-> None:
    x_train,x_test=train_test_split(df,random_state=42,test_size=test_size,shuffle=True)
    os.makedirs(os.path.dirname(train_path),exist_ok=True)
    x_train.to_csv(train_path,index=False)
    x_test.to_csv(test_path,index=False)
    
def main():

    test_size=reading_yml('params.yaml')
    print(test_size)
    df=load_data(r'data\raw\raw_data.csv')
    processed_data=data_preprocessing(df)
    print('dataset loaded')
    save_data(processed_data,train_path=r"data\processed\x_train.csv",test_path=r"data\processed\x_test.csv",test_size=test_size)
    print('saved train test dataset')
    
if __name__ == "__main__":
    main()