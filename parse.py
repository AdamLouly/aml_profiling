from azureml.core import Workspace, Experiment, Environment, ScriptRun
import mlflow
from mlflow import MlflowClient
from mlflow.entities import ViewType
import pandas as pd
import sys
import re
from utils import get_configs, get_env_variables, get_tags, get_prefix

client = mlflow.tracking.MlflowClient()

config = get_configs()
subscription_id = config["SUBSCRIPTION_ID"]
resource_group = config["RESOURCE_GROUP"]
workspace_name = config["WORKSPACE_NAME"]

workspace = Workspace(subscription_id, resource_group, workspace_name)

def parse_performance_metrics(data):
    # define the regular expression pattern to match the metrics
    pattern = r"Avg of.*?(\d+\.\d+) ms"

    # search for the pattern in the input data
    matches = re.findall(pattern, data)
    # extract the performance metrics if the pattern matched
    if matches:
        return ((matches[0], matches[1], matches[2], matches[3]),(matches[-4], matches[-3], matches[-2], matches[-1]))
    else:
        print("No performance metrics found.")
        return None

def parse_metric(log_str, metric_name):
    # Define the regular expression pattern
    if metric_name == "train_runtime":
        pattern = r"{}\s+=\s+([\d:]+\.\d+)".format(metric_name)
    elif metric_name == "eval_runtime":
        pattern = r"{}\s+=\s+([\d:]+\.\d+)".format(metric_name)
    else:
        pattern = r"{}\s+=\s+([\d.]+)".format(metric_name)
    
    # Search for the pattern in the log string
    match = re.search(pattern, log_str)
    
    if match:
        # Extract the metric value from the match object
        value_str = match.group(1)
        
        # Return the metric value as a tuple
        return value_str
    else:
        # If the pattern is not found, return None
        return "N/A"

def get_logs(portal_uri, model_name, deepspeed, run_type, experiment_name, run_id):

    current_experiment=dict(mlflow.get_experiment_by_name(experiment_name))
    experiment_id = current_experiment['experiment_id']

    # Retrive Portal URI for the run
    experiment = Experiment(workspace, experiment_name)

    # Get the script run using the experiment and run ID
    script_run = ScriptRun(experiment=experiment, run_id=run_id)
    portal_uri = script_run.get_portal_url()

    # Retrieve the logs from the run's artifacts directory
    logs_dir = f"runs:/{run_id}/train_logs"
    logs_artifacts = client.list_artifacts(run_id, "user_logs")
    logs_path = logs_artifacts[0].path
    logs_file = client.download_artifacts(run_id, logs_path)

    print("---- Downloaded Logs ----")

    with open(logs_file, 'r') as f:
        data = f.read()

    # Print the extracted information
    columns = [
        "portal_uri","model_name","deepspeed","run_type",
        "experiment_name", "run_id",
        "avg 2nd half", "FW+BW", "Optim", "Iter",
        "avg 2nd half_1", "FW+BW_1", "Optim_1", "Iter_1",
        "T-epoch", "T-loss", "T-rtime", "T-samples", "T-samp/s", "T-step/s",
        "E-epoch", "E-Acc", "E-loss", "E-rtime", "E-samples", "E-samp/s", "E-step/s"
    ]
    mappa = {}
    for col in columns:
        mappa[col] = []
    mappa["portal_uri"].append(portal_uri)
    mappa["model_name"].append(model_name)
    mappa["deepspeed"].append(deepspeed)
    mappa["run_type"].append(run_type)
    mappa["experiment_name"].append(experiment_name)
    mappa["run_id"].append(run_id)

    try:
        # Extract all the performance metrics
        performance_metrics_first, performance_metrics = parse_performance_metrics(data)
        # Performance metrics
        mappa["avg 2nd half"].append(performance_metrics[0])
        mappa["FW+BW"].append(performance_metrics[1])
        mappa["Optim"].append(performance_metrics[2])
        mappa["Iter"].append(performance_metrics[3])


        mappa["avg 2nd half_1"].append(performance_metrics_first[0])
        mappa["FW+BW_1"].append(performance_metrics_first[1])
        mappa["Optim_1"].append(performance_metrics_first[2])
        mappa["Iter_1"].append(performance_metrics_first[3])

    except:
        print("--- ERROR PULLING PERFORMANCE METRICS ---")
        mappa["avg 2nd half"].append("N/A")
        mappa["FW+BW"].append("N/A")
        mappa["Optim"].append("N/A")
        mappa["Iter"].append("N/A")

        mappa["avg 2nd half_1"].append("N/A")
        mappa["FW+BW_1"].append("N/A")
        mappa["Optim_1"].append("N/A")
        mappa["Iter_1"].append("N/A")

    # Train Metrics
    mappa["T-epoch"].append(parse_metric(data, "epoch"))
    mappa["T-loss"].append(parse_metric(data, "train_loss"))
    mappa["T-rtime"].append(parse_metric(data, "train_runtime"))
    mappa["T-samples"].append(parse_metric(data, "train_samples"))
    mappa["T-samp/s"].append(parse_metric(data, "train_samples_per_second"))
    mappa["T-step/s"].append(parse_metric(data, "train_steps_per_second"))

    # Eval Metrics
    mappa["E-epoch"].append(parse_metric(data, "epoch"))
    mappa["E-loss"].append(parse_metric(data, "eval_loss"))
    mappa["E-Acc"].append(parse_metric(data, "eval_accuracy"))
    mappa["E-rtime"].append(parse_metric(data, "eval_runtime"))
    mappa["E-samples"].append(parse_metric(data, "eval_samples"))
    mappa["E-samp/s"].append(parse_metric(data, "eval_samples_per_second"))
    mappa["E-step/s"].append(parse_metric(data, "eval_steps_per_second"))

    df = pd.DataFrame.from_dict(mappa)
    return df

def create_report(final_df, metric):      
    mappa = {}

    columns = ['model_name', 'deepspeed']
    run_types = final_df['run_type'].unique().tolist()

    for column in columns:
        mappa[column] = []

    for run_type in run_types:
        mappa[run_type] = []

    model_names = final_df['model_name'].unique().tolist()
    print("model names ", model_names)
    run_types = final_df['run_type'].unique().tolist()
    print("run types ", run_types)

    for model in model_names:

        mappa['model_name'].append(model)
        mappa['deepspeed'].append("NO")
        for run in run_types:
            temp = final_df[(final_df['model_name'] == model) & (final_df['run_type'] == run) & (final_df['deepspeed'] == 'NO')]
            try:
                mappa[run].append(float(temp[metric]))
            except:
                mappa[run].append('N/A')
                        
        mappa['model_name'].append(model)
        mappa['deepspeed'].append("YES")

        for run in run_types:
            # ds no
            temp = final_df[(final_df['model_name'] == model) & (final_df['run_type'] == run) & (final_df['deepspeed'] == 'YES')]
            try:
                mappa[run].append(float(temp[metric]))
            except:
                mappa[run].append('N/A')

    for key in mappa:
        print(key, len(mappa[key]))
    df = pd.DataFrame.from_dict(mappa)
    return df
    
if __name__ == "__main__":
    if len(sys.argv) > 1:
        filename = sys.argv[1]
    else:
        filename = "run_details.csv"

    # Read the Excel file into a pandas DataFrame
    df = pd.read_csv(filename, usecols=range(6))

    # Convert the DataFrame to an array of tuples
    data = [tuple(x) for x in df.to_numpy()]
    dfs = []

    # Print the array of tuples
    counter = 0
    for element in data:
        print(" Counting ..  ", counter)
        counter = counter + 1
        dfs.append(get_logs(*element))

    df = pd.concat(dfs)
    df_avg = create_report(df, 'avg 2nd half')
    df_tsamples = create_report(df, 'T-samp/s')
    merged_df = pd.merge(df_avg, df_tsamples, on=['model_name', 'deepspeed'], suffixes=('_avg 2nd half', '_training samples/s'))

    multiindex_columns = [
        ('model_name', ''),
        ('deepspeed', '')
    ]
    unique_run_types = df['run_type'].unique().tolist()
    for run_type in unique_run_types:
        multiindex_columns.append(('avg 2nd half', run_type))
        multiindex_columns.append(('training samples/s', run_type))

    merged_df.columns = pd.MultiIndex.from_tuples(multiindex_columns)

    output_file = filename.split(".")[0] + '.xlsx'
    # Save the DataFrames to an Excel file with different sheet names
    with pd.ExcelWriter(output_file) as writer:
        df.to_excel(writer, sheet_name='overall report')
        merged_df.to_excel(writer, sheet_name='report')
    
    print("Parsing completed, file saved : ", output_file)