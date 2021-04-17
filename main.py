# imports
import argparse
import os
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# huggingface imports
from transformers import BertForSequenceClassification
from transformers import BertTokenizer
from transformers import Trainer, TrainingArguments

# own imports
from data.LoadData import load_circa


def compute_metrics(pred):
    """
    Function that calculates the metrics that we report.
    Inputs:
        pred - Batch of predictions from the model
    """
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='micro')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }


def train_model(args):
    """
    Function for training and testing the BERT model.
    Inputs:
        args - Namespace object from the argument parser
    """

    # set the seed
    torch.manual_seed(args.seed)

    # check if GPU is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # print the model parameters
    print('-----TRAINING PARAMETERS-----')
    print('Model version: {}'.format(args.model_version))
    print('Labels: {}'.format(args.labels))
    print('PyTorch device: {}'.format(device))
    print('Num epochs: {}'.format(args.num_epochs))
    print('Learning rate: {}'.format(args.lr))
    print('Batch size: {}'.format(args.batch_size))
    print('Logging directory: {}'.format(args.log_dir))
    print('Results directory: {}'.format(args.results_dir))
    print('-----------------------------')

    # check how many labels to use
    if args.labels == 'strict':
        num_labels = 6
    else:
        num_labels = 4

    # load the model
    print('Loading model..')
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased',
        num_labels=num_labels
    ).to(device)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    print('Model loaded')

    # set the model to training mode
    model.train()

    # load the dataset
    print('Loading dataset..')
    train_set, dev_set, test_set = load_circa(args, tokenizer)
    print('Dataset loaded')

    # training arguments
    training_args = TrainingArguments(
        output_dir=args.results_dir + '/' + args.model_version + '_' + args.labels,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        lr_scheduler_type='constant',
        warmup_steps=50,
        logging_dir=args.log_dir + '/' + args.model_version + '_' + args.labels,
        evaluation_strategy='epoch',
        do_train =True,
        do_eval = True,
        gradient_accumulation_steps=2,
        eval_accumulation_steps=1,
        save_strategy='epoch',
        save_total_limit=1,
    )

    # create the trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_set,
        eval_dataset=dev_set,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    # train the model
    print('Starting training..')
    trainer.train()
    print('Training finished')

    # test the model
    print('Starting testing..')
    test_output = trainer.predict(test_set)
    print(test_output.metrics)
    print('Testing finished')


# command line arguments parsing
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # model hyperparameters
    parser.add_argument('--model_version', default='QA', type=str,
                        help='What model version to use. Default is QA (Question and Answer)',
                        choices=['QA', 'Q', 'A'])
    parser.add_argument('--labels', default='strict', type=str,
                        help='What labels to use. Default is strict',
                        choices=['strict', 'relaxed'])

    # training hyperparameters
    parser.add_argument('--num_epochs', default=3, type=int,
                        help='Number of epochs to train for. Default is 3')

    # optimizer hyperparameters
    parser.add_argument('--lr', default=3e-5, type=float,
                        help='Learning rate to use. Default is 3e-5')
    parser.add_argument('--batch_size', default=8, type=int,
                        help='Minibatch size. Default is 8')

    # loading hyperparameters
    parser.add_argument('--checkpoint_dir', default=None, type=str,
                        help='Directory where the model checkpoint is located. Default is None (no checkpoint used)')

    # other hyperparameters
    parser.add_argument('--seed', default=1234, type=int,
                        help='Seed to use for reproducing results. Default is 1234')
    parser.add_argument('--log_dir', default='./train_logs', type=str,
                        help='Directory where the training logs should be created. Default is ./train_logs')
    parser.add_argument('--results_dir', default='./results', type=str,
                        help='Directory where the training results should be created. Default is ./results')

    # parse the arguments
    args = parser.parse_args()

    # train the model
    train_model(args)
