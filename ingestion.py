import pandas as pd
import numpy as np
import os
import json
from datetime import datetime




#############Load config.json and get input and output paths
with open('config.json','r') as f:
    config = json.load(f) 

input_folder_path = config['input_folder_path']
output_folder_path = config['output_folder_path']



#############Function for data ingestion
def merge_multiple_dataframe():
    #check for datasets, compile them together, and write to an output file
    if not os.path.exists(output_folder_path):
        os.mkdir(output_folder_path)
    
    files = []
    for file in os.listdir(input_folder_path):
        if file.endswith('.csv'):
            files.append(file)
    
    df_list = []
    for filename in files:
        filepath = os.path.join(input_folder_path, filename)
        df = pd.read_csv(filepath)
        df_list.append(df)
    print(df_list[0].shape)
    print(len(df_list))
    print("names", df_list[0].columns)

    final_df = pd.concat(df_list)
    final_df.drop_duplicates(inplace=True)
    print(final_df.shape)
    print("names", final_df.columns)
    final_df.to_csv(os.path.join(output_folder_path, 'finaldata.csv'), index=False)

    with open(os.path.join(output_folder_path, 'ingestedfiles.txt'), 'w') as f:
        f.write('\n'.join(files))

if __name__ == '__main__':
    merge_multiple_dataframe()
