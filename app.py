from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import json
import os
import diagnostics
import scoring



######################Set up variables for use in our script
app = Flask(__name__)
app.secret_key = '1652d576-484a-49fd-913a-6879acfa6ba4'

with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 

prediction_model = None


#######################Prediction Endpoint
@app.route("/prediction", methods=['POST','OPTIONS'])
def predict():
    filepath = request.get_json()['filepath']

    prediction = diagnostics.model_predictions(filepath)

    return jsonify(prediction), 200

@app.route("/scoring", methods=['GET','OPTIONS'])
def score():        
    #check the score of the deployed model
    # Read the score from production deployment
    score_file = os.path.join(config['prod_deployment_path'], 'latestscore.txt')
    with open(score_file, 'r') as f:
        score_value = float(f.read())
    return jsonify(score_value), 200

#######################Summary Statistics Endpoint
@app.route("/summarystats", methods=['GET','OPTIONS'])
def summary_stats():        
    #check means, medians, and modes for each column
    stats = diagnostics.dataframe_summary()
    return jsonify(stats), 200

#######################Diagnostics Endpoint
@app.route("/diagnostics", methods=['GET','OPTIONS'])
def diagnose():
    timing = diagnostics.execution_time()
    missing = diagnostics.missing_data()
    dependencies = diagnostics.outdated_packages_list()
    
    return jsonify({
        'timing': timing,
        'missing_data': missing,
        'dependencies': dependencies
    }), 200

if __name__ == "__main__":    
    app.run(host='0.0.0.0', port=8000, debug=True, threaded=True)
