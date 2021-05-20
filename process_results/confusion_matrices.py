# imports
import numpy as np
import os
import ast

import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def get_results(filepath):
    """
    Reads the results from a Slurm output file. 
    Args:
    Inputs:
        filepath - Slurm output file from training.
    Outputs:
        results - A dictionary with results (confusion matrices). 
    """
    with open(filepath, "r") as a_file:
        for line in a_file:
            stripped_line = line.strip()
            if "Probably yes" in stripped_line:
                target_names = ast.literal_eval(stripped_line)
            if "confusion_matrix" in stripped_line:
                results = ast.literal_eval(stripped_line)
    results["target_names"] = target_names
    return results
                
                
def get_name_from_path(filepath):
    """
    Gets the correct name from a model slurm output.  
    Args:
    Inputs:
        filepath - Slurm output file from training.
    Outputs:
        name - the name of the model. 
    """
    filepath_list = filepath.split("/")
    
    for name in filepath_list:
        if "_advancedmetrics" in name:
            name = name.split("_")
            return name[0].capitalize()
    
    
def get_metrics(log_folder):
    """
    Goes through the folder of slurm output files, checks if there is 
    a slurm output file with confusion matrices.  
    Args:
    Inputs:
        log_folder - Folder containing slurm output files. 
    Outputs:
        metrics - a dict containing metrics for all models. 
    """

    metrics = {}

    pathlist = Path(log_folder).glob('**/*.out')

    for path in pathlist:
        filepath = str(path)
        if "advancedmetric" in filepath:
            name = get_name_from_path(filepath)
            results = get_results(filepath)
            metrics[name] = results
    return metrics


def plot_confusion_matrices(model, output_dir, normalize=True):
    """
    Plots the confusion matrices as described in Slurm output files.   
    Args:
    Inputs:
        model - The name of the model. 
        output_dir - Folder where the confusion matrices are saved to.
        normalize - Whether to normalize the matrices. 
    """

    results_list = metrics[model]["Circa"]["confusion_matrix"]
    target_names = list(metrics[model]["target_names"])
    df_cm = pd.DataFrame(results_list, index = target_names,
                      columns = target_names)
    
    if normalize:
        df_norm = df_cm.apply(lambda x:(x/x.sum()))

    plt.figure(figsize = (10,7))
    plt.title(model, fontsize=20)
    plt.gcf().subplots_adjust(left=0.34, bottom=0.34)

    sn.heatmap(df_norm, annot=True,  cmap="YlGnBu")
    plt.savefig('{}cm_{}.jpg'.format(output_dir, model))


if __name__ == '__main__':
    log_folder = "../training_logs"
    output_dir = "confusion_matrices/"

    # Create output directory.
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Read metric results from Slurm output files. 
    metrics = get_metrics(log_folder)

    # Plot the confusion matrices of each model. 
    for model_name in metrics.keys():
        plot_confusion_matrices(model_name, output_dir)


