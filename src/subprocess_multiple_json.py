# iterate over all json files in the ./json directory to run main.py
import os
import subprocess

def run_experiment_batch():
    json_dir = "./experiments/"
    json_files = [file for file in os.listdir(json_dir) if file.endswith(".json")]
    print(json_files)
    for json_file in json_files:
        json_path = os.path.join(json_dir, json_file)
        subprocess.run(["python", "main.py", json_path])
        os.remove(json_path)
run_experiment_batch()