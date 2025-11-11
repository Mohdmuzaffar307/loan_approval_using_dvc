import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.feature_selection import SelectKBest
import yaml

def load_yaml(path:str):
    feature_params=yaml.safe_load(open(path,'r'))
    return feature_params



def load_data(train_url:str,test_url:str )->tuple[pd.DataFrame,pd.DataFrame]:
    x_train=pd.read_csv(train_url)
    x_test=pd.read_csv(test_url)
    return x_train,x_test




def encoding(x_train :pd.DataFrame,x_test:pd.DataFrame)->tuple[pd.DataFrame,pd.DataFrame]:
    
    # Separate categorical and numerical columns
    cat_cols = x_train.select_dtypes(include=['object', 'category']).columns
    num_cols = x_train.select_dtypes(include=['int64', 'float64']).columns

    # One-hot encode categorical columns
    ohe = OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False)
    ohe.fit(x_train[cat_cols])
    

    x_train_cat = pd.DataFrame(ohe.transform(x_train[cat_cols]), 
                               columns=ohe.get_feature_names_out(cat_cols),
                               index=x_train.index)
    x_test_cat = pd.DataFrame(ohe.transform(x_test[cat_cols]), 
                              columns=ohe.get_feature_names_out(cat_cols),
                              index=x_test.index)
                              
   

    # Scale numerical columns
    scaler = StandardScaler()
    scaler.fit(x_train[num_cols])
    

    x_train_num = pd.DataFrame(scaler.transform(x_train[num_cols]), 
                               columns=num_cols,
                               index=x_train.index)
    x_test_num = pd.DataFrame(scaler.transform(x_test[num_cols]), 
                              columns=num_cols,
                              index=x_test.index)

    # Combine encoded categorical and scaled numerical columns
    x_train_final = pd.concat([x_train_num, x_train_cat], axis=1)
    x_test_final = pd.concat([x_test_num, x_test_cat], axis=1)

    return x_train_final, x_test_final




def feature_selection(x_train_final :pd.DataFrame,x_test_final :pd.DataFrame,k_value)->tuple[pd.DataFrame,pd.DataFrame]:
    select_feature=SelectKBest(k=k_value)
    select_feature.set_output(transform='pandas')
    select_feature.fit(x_train_final.drop(columns='loan_approved'),x_train_final.iloc[:,-1])
    
    x_train_final_feature=select_feature.transform(x_train_final.drop(columns='loan_approved'))
    x_test_final_feature=select_feature.transform(x_test_final.drop(columns='loan_approved'))
    
    return x_train_final_feature,x_test_final_feature



def save_data(train_path:str,test_path:str,x_train_final_feature:pd.DataFrame,x_test_final_feature:pd.DataFrame)->None:
    os.makedirs(os.path.dirname(train_path),exist_ok=True)
    os.makedirs(os.path.dirname(test_path),exist_ok=True)
    x_train_final_feature.to_csv(train_path,index=False)
    x_test_final_feature.to_csv(test_path,index=False)
    

def main():
    params_value=load_yaml('params.yaml')
    k_value=params_value['feature_engineering']['k']
    print(type(k_value))
    print('\n')
    x_train,x_test=load_data(train_url=r'data\processed\x_train.csv',test_url=r'data\processed\x_test.csv') 
    print('dataset_loaded')
    
    x_train_final,x_test_final=encoding(x_train,x_test)  
    print('encoding done')
    
    x_train_final_feature,x_test_final_feature=feature_selection(x_train_final,x_test_final,k_value)
    print("feature selection done")
    
    train_path=r"data\interim\x_train_final.csv"
    test_path=r"data\interim\x_test_final.csv"
    save_data(train_path,test_path,x_train_final_feature,x_test_final_feature)
    
if __name__ =="__main__":
    main()
    
