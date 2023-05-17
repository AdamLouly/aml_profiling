from azureml.core import Workspace, Experiment, Environment, ScriptRunConfig
from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core.compute_target import ComputeTargetException
from azureml.core.runconfig import MpiConfiguration
from azureml.core.runconfig import PyTorchConfiguration, DockerConfiguration
import argparse
import json
import re
import shutil
import os
import pandas as pd
from utils import get_configs, get_env_variables, get_tags, get_prefix

def submit_run(run_command, compute_target, experiment, myenv, pytorch_configuration, docker_config, tags, deepspeed=False):
    if deepspeed:
        shutil.copy(config["DS_CONFIG_PATH"], source_path)
        run_command += " --deepspeed " + config["DS_CONFIG_PATH"]
        tags["deepspeed"] = "YES"

    src = ScriptRunConfig(source_directory=source_path,
                          command=run_command,
                          compute_target=compute_target,
                          environment=myenv,
                          distributed_job_config=pytorch_configuration,
                          docker_runtime_config=docker_config)

    run = experiment.submit(src, tags=tags)

    mappa["portal_uri"].append(run.get_portal_url())
    mappa["model_name"].append(experiment_name)
    mappa["deepspeed"].append("YES" if deepspeed else "NO")
    mappa["run_type"].append(run_type)
    mappa["experiment_name"].append(experiment_name)
    mappa["run_id"].append(run.id)

    print("Jobs Submitted .. click here to check it in the portal:", run.get_portal_url())

# Parse command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--deepspeed', action='store_true', help='Run with DeepSpeed')
args = parser.parse_args()

config = get_configs()
subscription_id = config["SUBSCRIPTION_ID"]
resource_group = config["RESOURCE_GROUP"]
workspace_name = config["WORKSPACE_NAME"]

workspace = Workspace(subscription_id, resource_group, workspace_name)
compute_name = config["COMPUTE_NAME"]

compute_target = ComputeTarget(workspace=workspace, name=compute_name)

myenv = Environment(name=config["ENV_NAME"])
myenv.docker.base_image = config["DOCKER_BASE_IMAGE"]
myenv.python.user_managed_dependencies = True
myenv.python.interpreter_path = "/opt/conda/envs/ptca/bin/python"
myenv.environment_variables = get_env_variables()
docker_config = DockerConfiguration(use_docker=True, arguments=['--ipc=host'])

pytorch_configuration = PyTorchConfiguration(process_count=1, node_count=1)
run_type = config["RUN_TYPE"]

columns = ["portal_uri", "model_name", "deepspeed", "run_type", "experiment_name", "run_id"]
mappa = {col: [] for col in columns}

with open('models_details.json', 'r') as f:
    data = json.load(f)

for element in data:
    source_path = element["model_path"]

    model_name_pattern = r"--model_name_or_path\s+([\w\d\-_/]+)"
    match = re.search(model_name_pattern, element["command"])
    model_name_or_path = match.group(1)
    experiment_name = model_name_or_path.replace("/", "-")

    run_command = element["command"]
    experiment = Experiment(workspace, experiment_name)
    tags = get_tags()
    run_command = run_command

    submit_run(run_command, compute_target, experiment, myenv, pytorch_configuration, docker_config, tags, args.deepspeed)

# Create DataFrame and save results as CSV.
df = pd.DataFrame.from_dict(mappa)

prefix = get_prefix()
file_name = f"run_details_{prefix}.csv"

df.to_csv(file_name, index=False)

