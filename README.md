# CDM_Project
Project for Computational Dialogue Modelling (first year master AI @ UvA).

This repository contains research on the effect of Multi-Task Learning (MTL) on the domain of Indirect Answer Classification (IAC). The goal is to improve a BERT model on the IAC task by training the model in a multi-task fashion. We test a number of different tasks and datasets, but it is easy for further research to expand upon this research by adding other tasks or datasets.

## Content
This repository consists of the following key scripts:
* **main.py**: this is the main training loop for the model. This is the script that needs to be called when starting an experiment.
* **utils.py**: this script contains some helpful functions and takes away a lot of clutter from the main script.
* **multi_task/bert_mtl.py**: this script contains the Multi-Task adaption of the BERT model.
* **data/**: contains all dataloader classes for the different datasets.
* **data/multitask_dataloader**: script for combining multiple datasets into a single dataloader. Creates batches containing only a single task, but picks each batch from one of the datasets randomly.

## Prerequisites
* Anaconda. Available at: https://www.anaconda.com/distribution/

## Getting Started
1. Open Anaconda prompt and clone this repository (or download and unpack zip):
```bash
git clone https://github.com/Luuk99/CDM_Project.git
```
2. Create the environment:
```bash
conda env create -f environment.yml
```
3. Activate the environment:
```bash
conda activate CDM
```
4. Run the training script for the baseline model:
```bash
python main.py --progress_bar
```
Or provide a path to the checkpoint of an earlier run:
```bash
python main.py --checkpoint_path CHECKPOINT_PATH
```

## Replicating Results
TODO: add brief description on how to replicate the results

## Datasets
Most datasets are gathered from the [Huggingface](https://huggingface.co/) library and require no manual downloading. The following datasets are exceptions to this:
* For the IQAP dataset, download the zip from the [official website](http://compprag.christopherpotts.net/iqap.html) and move the csv to the **data/local_datasets/iqap** folder.

## Arguments
The models can be trained with the following command line arguments:
```bash
usage: main.py [-h] [--model_version MODEL_VERSION] [--labels LABELS] [--setting SETTING]
		    [--max_epochs MAX_EPOCHS] [--patience PATIENCE] [--lrs LRS] [--batch_size BATCH_SIZE]
		    [--aux_tasks AUX_TASKS] [--aux_probing] [--checkpoint_path CHECKPOINT_PATH]
		    [--seed SEED] [--results_dir RESULTS_DIR] [--progress_bar]

optional arguments:
  -h, --help            			Show help message and exit.
  --model_version MODEL_VERSION			What model version to use. Options: ['QA', 'Q', 'A']. Default is 'QA' (Question and Answer).
  --labels LABELS				What labels to use. Options: ['strict', 'relaxed']. Default is 'strict'.
  --setting SETTING				What test setting is used. Options: ['matched', 'unmatched']. Default is 'matched'.
  --max_epochs MAX_EPOCHS			Maximum number of epochs to train for. Default is 5.
  --patience PATIENCE				Stops training after patience number of epochs without improvement in development accuracy. Default is 3.
  --lrs LRS					Learning rates to use per task. Default is [3e-5] (for single task learning).
  --batch_size BATCH_SIZE			Minibatch size. Default is 8.
  --aux_tasks AUX_TASKS				Which auxiliary tasks to train on. Options: ['IQAP', 'SST2', 'MNLI', 'BOOLQ']. Default is [] (single task learning).
  --aux_probing 				Train only the classification layers for the auxiliary tasks.
  --checkpoint_path CHECKPOINT_PATH		Path to where the model checkpoint is located. Default is None (train from scratch).
  --seed SEED					Seed to use for reproducing results. Default is 1234.
  --results_dir RESULTS_DIR			Directory where the training results should be created. Default is './mtl_results'.
  --progress_bar				Use a progress bar indicator for interactive experimentation. Not to be used in conjuction with SLURM jobs.
```

## Errors
* **[Errno 2] No such file or directory: '..\multinli_1.0\multinli_1.0_dev_matched.jsonl'** when using MNLI as the auxilary task.
	* Download MultiNLI zip from the official website: https://cims.nyu.edu/~sbowman/multinli/
	* Extract the ZIP and place the **multinli_1.0** folder in the cache directory as provided by the error message

## Authors
* Luuk Kaandorp - luuk.kaandorp@student.uva.nl
* Casper Wortmann - casper.wortmann@student.uva.nl
* Damiaan Reijnaers - damiaan.reijnaers@student.uva.nl

## Acknowledgements
* Thanks to [Huggingface](https://huggingface.co/), the multi-task BERT model was adapted from their BERT implementations
