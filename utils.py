import os
import yaml
import datetime

CONFIG_FILE = 'config.yaml'

def get_prefix():
    """
    Returns the current day, month, hour, and minute as a formatted prefix.
    """
    current_datetime = datetime.datetime.now()
    formatted_datetime = current_datetime.strftime('%d%m_%H%M')
    return formatted_datetime

def get_configs():
    """
    Retrieve all the configurations from the config file.
    """
    with open(CONFIG_FILE, 'r') as file:
        config = yaml.safe_load(file)

    return config

def get_sub_configs():
    """
    Retrieve the sub-configurations from the config file.
    """
    config = get_configs()
    sub_configs = config.get("SUB_CONFIGS", [])
    return sub_configs

def get_env_variables(sub_config_name):
    """
    Retrieve environment variables from the config file for a specific sub-configuration.
    """
    sub_configs = get_sub_configs()
    for sub_config in sub_configs:
        if sub_config.get("NAME") == sub_config_name:
            env_variables = sub_config.get("ENVIRONMENT_VARIABLES", {})
            return env_variables
    return {}

def get_tags(sub_config_name):
    """
    Retrieve tags from the config file for a specific sub-configuration.
    """
    sub_configs = get_sub_configs()
    for sub_config in sub_configs:
        if sub_config.get("NAME") == sub_config_name:
            tags = sub_config.get("TAGS", {})
            return tags
    return {}

# Example usage
prefix = get_prefix()
sub_configs = get_sub_configs()
env_variables = get_env_variables("SubConfig1")
tags = get_tags("SubConfig1")

print(f"Prefix: {prefix}")
print("Sub-configurations:")
for sub_config in sub_configs:
    name = sub_config.get("NAME")
    run_type = sub_config.get("RUN_TYPE")
    envs = get_env_variables(name)
    print(f"envs: {envs}")

    print(f"Name: {name}")
    print(f"Run type: {run_type}")

print("Environment variables:")
print(env_variables)
print("Tags:")
print(tags)
