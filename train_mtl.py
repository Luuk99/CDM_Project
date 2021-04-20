# imports
import argparse
import os
import json
import random
from timeit import default_timer as timer
import datetime
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# huggingface imports
from transformers import BertForSequenceClassification
from transformers import BertTokenizer
from transformers import AdamW

# own imports
from data.LoadData import load_mtl_data
from multi_task.bert_mtl import MLTBertForSequenceClassification


def perform_epoch(model, optimizer, batch_size, dataset, device, train=True):
    """
    Function that performs an epoch for the given model.
    Inputs:
        model - BERT model instance
        optimizer - AdamW optimizer instance
        batch_size - Size of the mini batches
        dataset - Dataset to use
        device - PyTorch device to use
        train - Whether to train or test the model
    Outputs:
        mean_loss - Mean loss over the epoch
        accuracy - Accuracy over all instances in the dataset
    """

    # set model to training or evaluation
    if train:
        model.train()
    else:
        model.eval()

    # start a timer for the epoch time
    start_time = timer()

    # initialize lists for the losses, predictions and labels
    predictions = []
    labels = []
    losses = []

    # shuffle the datasets
    dataset = dataset.shuffle(load_from_cache_file=False)
    batches = [range(len(dataset))[i * batch_size:(i + 1) * batch_size] for i in range((len(dataset) + batch_size - 1) // batch_size )]

    # loop over the batches
    for step, batch in enumerate(batches):
        model.zero_grad()

        # get the batch from the dataset
        batch = dataset[batch[0]:batch[-1]]
        input_ids = torch.tensor(batch['input_ids'], device=device)
        attention_mask = torch.tensor(batch['attention_mask'], device=device)
        batch_labels = torch.tensor(batch['labels'], device=device)
        batch_task = torch.tensor(batch['task_idx'], device=device)

        # pass the batch through the model
        outputs = model(input_ids, attention_mask=attention_mask, labels=batch_labels, task_idx=batch_task)
        loss = outputs.loss

        if train:
            # optimize the loss
            loss.backward()
            optimizer.step()

        # add the loss, label and prediction to the lists
        losses.append(loss)
        predictions.append(outputs.logits)
        labels.append(batch_labels)

    # average the training loss for the epoch
    mean_loss = torch.mean(torch.stack(losses, dim=0), dim=0)

    # calculate the training accuracy
    predictions = torch.argmax(torch.cat(predictions, dim=0), dim=1)
    labels = torch.cat(labels, dim=0)
    accuracy = torch.true_divide(torch.sum(predictions == labels), torch.tensor(labels.shape[0], device=labels.device))

    # record the end time
    end_time = timer()

    # calculate the elapsed time
    elapsed_time = str(datetime.timedelta(seconds=(end_time - start_time)))

    # return the elapsed time, loss and accuracy
    return elapsed_time, mean_loss, accuracy


def handle_matched(args, device):
    """
    Function handling the matched setting.
    Inputs:
        args - Namespace object from the argument parser
        device - PyTorch device to use
    """

    # check how many labels to use
    if args.labels == 'strict':
        num_labels = 6
    else:
        num_labels = 4

    # load the model
    print('Loading model..')
    model = MLTBertForSequenceClassification.from_pretrained('bert-base-uncased',
        num_labels=num_labels
    ).to(device)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # add the auxilary tasks
    model.add_aux_classifiers([6, 6])

    # create the optimizer
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr)
    print('Model loaded')

    # set the model to training mode
    model.train()

    # load the dataset
    print('Loading dataset..')
    train_set, dev_set, test_set = load_mtl_data(args, tokenizer)
    print('Dataset loaded')

    # train the model
    print('Starting training..')
    for epoch in range(1, args.num_epochs + 1):
        # perform a training epoch
        train_time, train_loss, train_acc = perform_epoch(model, optimizer, args.batch_size, train_set, device, train=True)

        # perform a development epoch
        with torch.no_grad():
            dev_time, dev_loss, dev_acc = perform_epoch(model, optimizer, args.batch_size, dev_set, device, train=False)

        # print the epoch measures
        print('Epoch {}: train_time {} & train_loss {} & train_acc {} & dev_time {} & dev_loss {} & dev_acc {}'.format(
                epoch,
                train_time,
                round(train_loss.item(), 4),
                round(train_acc.item(), 4),
                dev_time,
                round(dev_loss.item(), 4),
                round(dev_acc.item(), 4),
             )
         )
    print('Training finished')

    # test the model
    print('Starting testing..')
    with torch.no_grad():
        test_time, test_loss, test_acc = perform_epoch(model, optimizer, args.batch_size, test_set, device, train=False)
    print('Testing: test_time {} & test_loss {} & test_acc {}'.format(
            test_time,
            round(test_loss.item(), 4),
            round(test_acc.item(), 4),
         )
     )



def handle_unmatched(args, device):
    """
    Function handling the unmatched setting.
    Inputs:
        args - Namespace object from the argument parser
        device - PyTorch device to use
    """

    # create a dictionary for the results of the different test scenarios
    test_results = dict()

    # list of the scenarios
    scenarios = ["X wants to know about Y's food preferences.",
        "X wants to know what activities Y likes to do during weekends.",
        "X wants to know what sorts of books Y likes to read.",
        "Y has just moved into a neighbourhood and meets his/her new neighbour X.",
        "X and Y are colleagues who are leaving work on a Friday at the same time.",
        "X wants to know about Y's music preferences.",
        "Y has just travelled from a different city to meet X.",
        "X and Y are childhood neighbours who unexpectedly run into each other at a cafe.",
        "Y has just told X that he/she is thinking of buying a flat in New York.",
        "Y has just told X that he/she is considering switching his/her job."
    ]

    # load the tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # check how many labels to use
    if args.labels == 'strict':
        num_labels = 6
    else:
        num_labels = 4

    # train and test the model for each held-out scenario
    for index, test_scenario in enumerate(scenarios):
        # pick a random development scenario
        left_scenarios = scenarios[:]
        left_scenarios.remove(test_scenario)
        dev_scenario = random.choice(left_scenarios)

        # TODO: add this
        exit()


def train_model(args):
    """
    Function for training and testing the BERT model.
    Inputs:
        args - Namespace object from the argument parser
    """

    # set the seed
    #torch.manual_seed(args.seed)

    # check if GPU is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # print the model parameters
    print('-----TRAINING PARAMETERS-----')
    print('Model version: {}'.format(args.model_version))
    print('Labels: {}'.format(args.labels))
    print('Setting: {}'.format(args.setting))
    print('PyTorch device: {}'.format(device))
    print('Num epochs: {}'.format(args.num_epochs))
    print('Learning rate: {}'.format(args.lr))
    print('Batch size: {}'.format(args.batch_size))
    print('Logging directory: {}'.format(args.log_dir))
    print('Results directory: {}'.format(args.results_dir))
    print('-----------------------------')

    # check which setting is selected
    if args.setting == 'matched':
        handle_matched(args, device)
    else:
        handle_unmatched(args, device)


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
    parser.add_argument('--setting', default='matched', type=str,
                        help='What test setting is used. Default is matched',
                        choices=['matched', 'unmatched'])

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
    parser.add_argument('--log_dir', default='./mtl_train_logs', type=str,
                        help='Directory where the training logs should be created. Default is ./mtl_train_logs')
    parser.add_argument('--results_dir', default='./mtl_results', type=str,
                        help='Directory where the training results should be created. Default is ./mtl_results')

    # parse the arguments
    args = parser.parse_args()

    # train the model
    train_model(args)
