import pandas as pd
import numpy as np
import timeit
import os
import json
import subprocess
import pickle

##################Load config.json and get environment variables
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
test_data_path = os.path.join(config['test_data_path'])
prod_deployment_path = os.path.join(config['prod_deployment_path'])
model_path = os.path.join(config['output_model_path']) 

##################Function to get model predictions
def model_predictions(dataset_path=None):
    if dataset_path is None:
        dataset_path = os.path.join(test_data_path, 'testdata.csv')
    
    df = pd.read_csv(dataset_path)
    X = df[['lastmonth_activity', 'lastyear_activity', 'number_of_employees']]

    model_file = os.path.join(prod_deployment_path, 'trainedmodel.pkl')
    with open(model_file, 'rb') as f:
        model = pickle.load(f)
        
    model_predictions = model.predict(X)
    
    #read the deployed model and a test dataset, calculate predictions
    return model_predictions.tolist() #return value should be a list containing all predictions

##################Function to get summary statistics
def dataframe_summary():
    data_file = os.path.join(dataset_csv_path, 'finaldata.csv')
    df = pd.read_csv(data_file)

    numerical_columns = ['lastmonth_activity', 'lastyear_activity', 'number_of_employees']
    summary_statistics = []
    for col in numerical_columns:
        mean = df[col].mean()
        median = df[col].median()
        std = df[col].std()
        summary_statistics.extend([mean, median, std])
    return summary_statistics

##################Function to check for missing data
def missing_data():
    data_file = os.path.join(dataset_csv_path, 'finaldata.csv')
    df = pd.read_csv(data_file)
    
    na_percentages = []
    for col in df.columns:
        na_count = df[col].isna().sum()
        na_percent = (na_count / len(df)) * 100
        na_percentages.append(na_percent)
    
    return na_percentages

##################Function to get timings
def execution_time():
    # Time data ingestion
    start_time = timeit.default_timer()
    os.system('python ingestion.py')
    ingestion_time = timeit.default_timer() - start_time
    
    # Time model training
    start_time = timeit.default_timer()
    os.system('python training.py')
    training_time = timeit.default_timer() - start_time
    
    return [ingestion_time, training_time]

##################Function to check dependencies
def outdated_packages_list():
    outdated = subprocess.check_output(['pip', 'list', '--outdated'])
    outdated = outdated.decode('utf-8')

    return outdated


if __name__ == '__main__':
    print("Model Predictions:", model_predictions())
    print("Summary Statistics:", dataframe_summary())
    print("Missing Data %:", missing_data())
    print("Execution Times:", execution_time())
    print("Outdated Packages:\n", outdated_packages_list())





    
