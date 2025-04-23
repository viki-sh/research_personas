# %%
import json
import os
from openai import OpenAI
from dotenv import load_dotenv
from pathlib import Path

# %%
import os

with open('./dsl_project.zshrc', 'r') as file:
    api_key = file.read().strip()
    os.environ['OPENAI_API_KEY'] = api_key

print(f"API key set: {os.environ['OPENAI_API_KEY'][:5]}...")  # Shows first 5 chars for verification


# %%
#this can't be pushed to git
#os.environ.get("OPENAI_API_KEY")

# %%
load_dotenv()
api_key = os.environ.get("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)


# %%
# response = client.responses.create(
#   model="gpt-4.1-nano",
#   input="Tell me a three sentence bedtime story about a unicorn."
# )
# response

# %%
# Import the ScenarioGenerator class
from generation import ScenarioGenerator
from generation import run_agents_2_and_3

# %%
# Configure the generator
config = {
    "embedding_model": "all-MiniLM-L6-v2",  # Lightweight model, good for 16GB RAM
    "similarity_threshold": 0.85,  # Adjust based on desired uniqueness
    "batch_size": 2,  # Number of scenarios to generate in parallel
    "max_workers": 4,  # Set to half your CPU count for optimal performance
    "output_dir": "output",
    "checkpoint_interval": 2,  # Save progress every 25 unique scenarios
    #"llm_provider": "openai",  # Change to "openai" if preferred
    "personas": 'k12_combinations.json'
}


# %%

# Usage in notebook
generator = ScenarioGenerator(config)
generator.run_batch(1)
# Analyze the results
#generator.analyze_results()

# # Restore original method if needed
# # generator.generate_scenario_text = original_method

# %%

# Find latest JSON file
json_files = list(Path("output").glob("scenarios_*.json"))
if not json_files:
    print("no file found")
        
latest_json = max(json_files, key=lambda x: x.stat().st_mtime)
timestamp = "_".join(latest_json.stem.split('_')[1:])

run_agents_2_and_3(latest_json, timestamp)

