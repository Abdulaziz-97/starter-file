
import pandas as pd
import numpy as np
import timeit
import os
import json
import subprocess

##################Load config.json and get environment variables
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
test_data_path = os.path.join(config['test_data_path']) 
model_path = os.path.join(config['output_model_path']) 
##################Function to get model predictions
def model_predictions():
    if dataset_path is None:
        dataset_path = os.path.join(dataset_csv_path, 'finaldata.csv')
    
    df = pd.read_csv(dataset_path)
    X = df[['lastmonth_activity', 'lastyear_activity', 'number_of_employees']]

    model_file = os.path.join(model_path, 'trainedmodel.pkl')
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
    model_predictions()
    dataframe_summary()
    execution_time()
    outdated_packages_list()





    
