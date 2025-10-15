import pickle
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os



###############Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 




##############Function for reporting
def score_model():
    # Load test data
    test_file = os.path.join(dataset_csv_path, 'testdata.csv')
    test_df = pd.read_csv(test_file)
    X_test = test_df[['lastmonth_activity', 'lastyear_activity', 
                      'number_of_employees']]
    y_test = test_df['exited']
    
    # Load deployed model
    model_file = os.path.join(model_path, 'trainedmodel.pkl')
    with open(model_file, 'rb') as f:
        model = pickle.load(f)
    
    # Get predictions
    y_pred = model.predict(X_test)
    
    # Create confusion matrix
    cm = metrics.confusion_matrix(y_test, y_pred)
    
    # Plot it
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    
    # Save plot
    plot_file = os.path.join(config['output_model_path'], 'confusionmatrix.png')
    plt.savefig(plot_file)
    plt.close()
    
    print(f"Confusion matrix saved to {plot_file}")
if __name__ == '__main__':
    score_model()
