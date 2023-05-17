import os
import yaml
import datetime

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
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    return config

def get_env_variables():
    """
    Retrieve environment variables from the config file.
    """
    config = get_configs()
    env_variables = config.get("ENVIRONMENT_VARIABLES", {})
    return env_variables

def get_tags():
    """
    Retrieve tags from the config file.
    """
    config = get_configs()
    tags = config.get("TAGS", {})
    return tags
