import requests
import json
import os

with open('config.json', 'r') as f:
    config = json.load(f)

# URL of your Flask app
URL = "http://127.0.0.1:8000"

# Call all endpoints
responses = []

# 1. Predictions
response1 = requests.post(
    f"{URL}/prediction",
    json={'filepath': 'testdata/testdata.csv'}
)
responses.append("Predictions:\n" + response1.text)

# 2. Scoring
response2 = requests.get(f"{URL}/scoring")
responses.append("Scoring:\n" + response2.text)

# 3. Summary Stats
response3 = requests.get(f"{URL}/summarystats")
responses.append("Summary Statistics:\n" + response3.text)

# 4. Diagnostics
response4 = requests.get(f"{URL}/diagnostics")
responses.append("Diagnostics:\n" + response4.text)

# Save all responses
output_file = os.path.join(config['output_model_path'], 'apireturns.txt')
with open(output_file, 'w') as f:
    f.write('\n\n'.join(responses))

print(f"API responses saved to {output_file}")