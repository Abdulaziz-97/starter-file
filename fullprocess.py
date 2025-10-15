import training
import scoring
import deployment
import diagnostics
import reporting
import ingestion
import os
import json

with open('config.json', 'r') as f:
    config = json.load(f)

input_folder_path = config['input_folder_path']
prod_deployment_path = config['prod_deployment_path']

##################Check and read new data
print("=" * 50)
print("STEP 1: Checking for new data...")
print("=" * 50)

# First, read ingestedfiles.txt from production deployment
ingested_files_path = os.path.join(prod_deployment_path, 'ingestedfiles.txt')

with open(ingested_files_path, 'r') as f:
    ingested_files = f.read().splitlines()

print(f"Previously ingested files: {ingested_files}")

# Second, determine whether the source data folder has files that aren't listed in ingestedfiles.txt
source_files = []
for file in os.listdir(input_folder_path):
    if file.endswith('.csv'):
        source_files.append(file)

print(f"Current source files: {source_files}")

# Find new files
new_files = [f for f in source_files if f not in ingested_files]

##################Deciding whether to proceed, part 1
if len(new_files) == 0:
    print("✓ No new data found. Exiting.")
    exit()

print(f"✓ New data found: {new_files}")

##################Ingest new data
print("\n" + "=" * 50)
print("STEP 2: Ingesting new data...")
print("=" * 50)
ingestion.merge_multiple_dataframe()

##################Checking for model drift
print("\n" + "=" * 50)
print("STEP 3: Checking for model drift...")
print("=" * 50)

# Get old score from production
old_score_path = os.path.join(prod_deployment_path, 'latestscore.txt')
with open(old_score_path, 'r') as f:
    old_score = float(f.read())

print(f"Old production score: {old_score}")

# Get new score with latest data
new_score = scoring.score_model()
print(f"New score with latest data: {new_score}")

##################Deciding whether to proceed, part 2
# If new score is >= old score, no drift (model is still good or better)
if new_score >= old_score:
    print("✓ No model drift detected (new score >= old score). Exiting.")
    exit()

print(f"✗ Model drift detected! (new score {new_score} < old score {old_score})")

##################Re-training
print("\n" + "=" * 50)
print("STEP 4: Re-training model...")
print("=" * 50)
training.train_model()

##################Re-scoring
print("\n" + "=" * 50)
print("STEP 5: Scoring new model...")
print("=" * 50)
scoring.score_model()

##################Re-deployment
print("\n" + "=" * 50)
print("STEP 6: Deploying new model...")
print("=" * 50)
deployment.store_model_into_pickle()

##################Diagnostics and reporting
print("\n" + "=" * 50)
print("STEP 7: Running diagnostics and reporting...")
print("=" * 50)
reporting.score_model()

# Note: To run apicalls.py, you need the API running separately
# os.system('python apicalls.py')

print("\n" + "=" * 50)
print("✓ FULL PROCESS COMPLETE!")
print("=" * 50)



