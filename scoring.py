from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import os
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import json



#################Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
test_data_path = os.path.join(config['test_data_path']) 


#################Function for model scoring
def score_model():
    #this function should take a trained model, load test data, and calculate an F1 score for the model relative to the test data
    #it should write the result to the latestscore.txt file
    test_file = os.path.join(test_data_path, 'testdata.csv')
    test_df = pd.read_csv(test_file)
    X_test = test_df[['lastmonth_activity', 'lastyear_activity', 'number_of_employees']]
    y_test = test_df['exited']
   
    model_file = os.path.join(model_path, 'trainedmodel.pkl')
    with open(model_file, 'rb') as f:
        model = pickle.load(f)
    
    y_pred = model.predict(X_test)
    f1_score = metrics.f1_score(y_test, y_pred)
    score_file = os.path.join(model_path, 'latestscore.txt')
    with open(score_file, 'w') as f:
        f.write(str(f1_score))
    print("F1 score written to", score_file)
    return f1_score