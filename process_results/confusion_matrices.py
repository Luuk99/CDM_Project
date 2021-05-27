# imports
import numpy as np
import os
import ast
import argparse
import seaborn as sn
import pandas as pd
from pathlib import Path


# Setting up your environment for exporting a pdf can be a bit of a hassle. 
# If these settings raise an error, import the pdf backand by installing mactex: 
# brew install --cask mactex 
import matplotlib as mpl
import os
os.environ["PATH"] += os.pathsep + '/usr/local/texlive/2021/bin/universal-darwin'
mpl.use('pgf')
mpl.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'font.size': 15,
    'text.usetex': True,
    'pgf.rcfonts': False,
})
import matplotlib.pyplot as plt

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
            if "yes nor no" in stripped_line:
                target_names = ast.literal_eval(stripped_line)
                if "Circa" in target_names.keys():
                    target_names = target_names["Circa"]

            if "confusion_matrix" in stripped_line:
                results = ast.literal_eval(stripped_line)
    results["target_names"] = target_names
    return results
                

def check_for_metrics(filepath):
    """
    Checks wheter a file contains metrics.  
    Args:
    Inputs:
        filepath - Slurm output file from training.
    Outputs:
        The name of the model. 
    """
    with open(filepath) as f:
        if 'Advanced metrics: True' in f.read():
            filepath_list = filepath.split("/")
            return filepath_list[-2].replace('_advancedmetrics', '')
        else:
            return None
  
    
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
        name = check_for_metrics(filepath)
        if name:
            results = get_results(filepath)
            metrics[name] = results
    return metrics

def get_shorter_name(input_names):
    """
    The confusion matrices will get big if the provided names are used,
    this functions returns shorter names for a target name. 
    Args:
    Inputs:
        input_names: the input name, a rather long string.
    Outputs:
        input_names: shortened input names.
    """
    input_names = list(dict(sorted(input_names.items(), key=lambda item: item[1])).keys())

    for idx, name in enumerate(input_names):
        if name == 'Probably yes / sometimes yes':
            input_names[idx] = "P.Yes"
        elif name == 'Yes, subject to some conditions':
            input_names[idx] = "C.Yes"
        elif name == 'Probably no':
            input_names[idx] = "P.No"
        elif name == 'In the middle, neither yes nor no':
            input_names[idx] = "Mid"
    return input_names


def plot_confusion_matrices(model, output_dir, normalize=True):
    """
    Plots the confusion matrices as described in Slurm output files.   
    Args:
    Inputs:
        model - The name of the model. 
        output_dir - Folder where the confusion matrices are saved to.
        normalize - Whether to normalize the matrices. 
    """
    print("plotting for model: ", model)
    model_title = model.split("_")[0].capitalize()
    results_list = metrics[model]["Circa"]["confusion_matrix"]
    target_names = metrics[model]["target_names"]
    target_names = get_shorter_name(target_names)

    df_cm = pd.DataFrame(results_list, index = target_names,
                      columns = target_names)
    
    if normalize:
        df_norm = df_cm.apply(lambda x:(x/x.sum()))

    plt.figure(figsize = (5, 5))
    plt.title(model_title, fontsize=26, fontweight="bold")
    sn.heatmap(df_norm, annot=True,  cmap="YlGnBu", cbar=False)
    plt.tight_layout()
    if args.output_type == "pdf":
        plt.savefig('{}cm_{}.pdf'.format(output_dir, model), format='pdf')
    else:
        plt.savefig('{}cm_{}.jpg'.format(output_dir, model))

if __name__ == '__main__':
    # Analyse arguments from commandline.
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--output_type', default='jpg', type=str,
            help='What output type the confusion matrix should be. Select pdf for vectorized output type.'
            'Setting up your environment pdf can be a bit of a hassle.',
            choices=['jpg', 'pdf'])
    parser.add_argument('--log_folder', default='../training_logs', type=str,
            help='Directory where the training results are located.')
    parser.add_argument('--output_dir', default='confusion_matrices/', type=str,
            help='Directory where the confusion matrices should be stored.')

    # Parse the arguments.
    args = parser.parse_args()

    # Create output directory.
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Read metric results from Slurm output files. 
    metrics = get_metrics(args.log_folder)

    # Plot the confusion matrices of each model. 
    for model_name in metrics.keys():
        plot_confusion_matrices(model_name, args.output_dir)


