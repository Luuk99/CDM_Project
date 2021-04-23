# CDM_Project
Project for Computational Dialogue Modelling (first year master AI @ UvA).

## Content
TODO: add brief description of the content

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
TODO: add all steps to getting started

## Replicating Results
TODO: add brief description on how to replicate the results

## Arguments
The models can be trained with the following command line arguments:
```bash
usage: main.py [-h] [--model MODEL] [--lr LR] [--lr_decay LR_DECAY]
		    [--lr_decrease_factor LR_DECREASE_FACTOR] [--lr_threshold LR_THRESHOLD] 
		    [--batch_size BATCH_SIZE] [--checkpoint_dir CHECKPOINT_DIR]
		    [--seed SEED] [--log_dir LOG_DIR] [--progress_bar] [--development]

optional arguments:
  -h, --help            			Show help message and exit.
  --model MODEL					What model to use. Options: ['AWE', 'UniLSTM', 'BiLSTM', 'BiLSTMMax']. Default is 'AWE'.
  --lr LR					Learning rate to use. Default is 0.1.
  --lr_decay LR_DECAY				Learning rate decay after each epoch. Default is 0.99.
  --lr_decrease_factor LR_DECREASE_FACTOR	Factor to divide learning rate by when dev accuracy decreases. Default is 5.
  --lr_threshold LR_THRESHOLD			Learning rate threshold to stop at. Default is 10e-5.
  --batch_size BATCH_SIZE			Minibatch size. Default is 64.
  --checkpoint_dir CHECKPOINT_DIR		Directory where the pretrained model checkpoint is located. Default is None (no checkpoint used).
  --seed SEED					Seed to use for reproducing results. Default is 1234.
  --log_dir LOG_DIR				Directory where the PyTorch Lightning logs should be created. Default is 'pl_logs'.
  --progress_bar				Use a progress bar indicator for interactive experimentation. Not to be used in conjuction with SLURM jobs.
  --development					Limit the size of the datasets in development.
```

## Errors
* **[Errno 2] No such file or directory: '..\multinli_1.0\multinli_1.0_dev_matched.jsonl'** when using MNLI as the auxilary task.
	* Download MultiNLI zip from the official website: https://cims.nyu.edu/~sbowman/multinli/
	* Extract the ZIP and place the **multinli_1.0** folder in the cache directory as provided by the error message

## Authors
* Luuk Kaandorp - luuk.kaandorp@student.uva.nl
* Casper
* Damiaan 

## Acknowledgements
* Circa dataset was cloned from the [original project GitHub](https://github.com/google-research-datasets/circa).