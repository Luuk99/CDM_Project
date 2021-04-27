# imports
import argparse
import os
import json
import random
from timeit import default_timer as timer
import datetime
import torch
from tqdm import tqdm

# own imports
from data.load_circa_data import LoadCircaMatched, LoadCircaUnmatched
from data.load_sst2_data import LoadSST2
from data.load_mnli_data import LoadMNLI
from data.load_boolq_data import LoadBoolQ
from data.load_iqap_data import LoadIQAP
from data.multitask_dataloader import MultiTaskDataloader
from utils import create_dataloader, handle_epoch_metrics, create_path, initialize_model

# set Huggingface logging to error only
import transformers
transformers.logging.set_verbosity_error()


def perform_step(model, optimizer, batch, device, task_idx, train=True):
    """
    Function that performs an epoch for the given model and task.
    Inputs:
        model - BERT model instance
        optimizer - AdamW optimizer instance for the given task
        batch - Batch from the dataset to use in the step
        device - PyTorch device to use
        task_idx - Index of the task
        train - Whether to train or test the model
    Outputs:
        loss - Loss of the step
        logits - Predictions of the model
        batch_labels - Real labels of the batch
    """

    # get the features of the batch
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    batch_labels = batch['labels'].to(device)
    token_type_ids = batch['token_type_ids'].to(device)

    # pass the batch through the model
    outputs = model(input_ids, attention_mask=attention_mask, labels=batch_labels, token_type_ids=token_type_ids, task_idx=task_idx)
    loss = outputs.loss

    if train:
        # backward using the loss
        loss.backward()

        # clip the gradient norm
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)

        # set a step with the optimizer
        optimizer.step()
        optimizer.zero_grad()

    # return the loss, label and prediction
    return loss, outputs.logits, batch_labels


def perform_epoch(args, model, optimizers, dataset, device, train=True):
    """
    Function that performs an epoch for the given model.
    Inputs:
        args - Namespace object from the argument parser
        model - BERT model instance
        optimizers - List of optimizers to use
        dataset - Dataset to use
        device - PyTorch device to use
        train - Whether to train or test the model
    Outputs:
        epoch_results - Dictionary containing the average epoch results
    """

    # set model to training or evaluation
    if train:
        model.train()
    else:
        model.eval()

    # start a timer for the epoch time
    start_time = timer()

    # initialize dictionary for the results
    result_dict = {}

    # loop over the batches
    if (args.progress_bar):
        dataset = tqdm(dataset)
    for (task_name, task_idx, batch) in dataset:
        # perform a step for the main task
        step_loss, step_predictions, step_labels = perform_step(model, optimizers[task_idx], batch, device, task_idx, train)

        # add the results to the dictionary
        if task_name in result_dict:
            result_dict[task_name]['predictions'].append(step_predictions.squeeze())
            result_dict[task_name]['labels'].append(step_labels.squeeze())
            result_dict[task_name]['losses'].append(step_loss)
        else:
            result_dict[task_name] = {
                'predictions': [step_predictions.squeeze()],
                'labels': [step_labels.squeeze()],
                'losses': [step_loss]
            }

    # calculate the loss and accuracy for the different tasks
    epoch_results = handle_epoch_metrics(result_dict)

    # record the end time
    end_time = timer()

    # calculate the elapsed time
    elapsed_time = str(datetime.timedelta(seconds=(end_time - start_time)))

    # add the time to the epoch results
    epoch_results['time'] = {'elapsed_time': elapsed_time}

    # return the epoch results
    return epoch_results


def train_model(args, model, optimizers, train_set, dev_set, test_set, device, path):
    """
    Function for training the model.
    Inputs:
        args - Namespace object from the argument parser
        model - BERT model instance
        optimizers - List of optimizers to use
        train_set - Multitask training set
        dev_set - Multitask development set
        test_set - Multitask test set
        device - PyTorch device to use
        path - Path for storing the results
    Outputs:
        model - Trained BERT model instance
        main_optimizer - AdamW optimizer instance for the main task
        aux_optimizer - AdamW optimizer instance for auxilary tasks
        gathered_results - Measures of the training process
    """

    print('Starting training..')
    gathered_results = {}

    # evaluate the model before training
    print('Epoch 0:')
    with torch.no_grad():
        dev_results = perform_epoch(args, model, optimizers, dev_set, device, train=False)
    print('Dev results:')
    print(dev_results)

    # save the pre-training evaluation measures
    gathered_results['epoch0'] = {'dev': dev_results}

    # train the model
    best_dev_acc = 0
    epochs_no_improvement = 0
    for epoch in range(1, args.max_epochs + 1):
        print('Epoch {}:'.format(epoch))

        # perform a training epoch
        train_results = perform_epoch(args, model, optimizers, train_set, device, train=True)

        # perform a development epoch
        with torch.no_grad():
            dev_results = perform_epoch(args, model, optimizers, dev_set, device, train=False)

        # print the epoch measures
        print('Train results:')
        print(train_results)
        print('Dev results:')
        print(dev_results)

        # save the epoch measures
        gathered_results['epoch' + str(epoch)] = {'train' : train_results, 'dev': dev_results}

        # check whether to save the model or not
        if (round(dev_results['Circa']['accuracy'], 2) > best_dev_acc):
            epochs_no_improvement = 0
            best_dev_acc = round(dev_results['Circa']['accuracy'], 2)
            print('Saving new best model..')
            torch.save({
                'epoch': epoch,
                'bert_state_dict': model.bert.state_dict(),
                'optimizer_state_dicts': [optimizer.state_dict() for optimizer in optimizers],
            }, os.path.join(path, "best_model.pt"))
            print('New best model saved')
        else:
            epochs_no_improvement += 1
            if epochs_no_improvement == args.patience:
                print('---')
                break

        print('---')
    print('Training finished')

    # load the best checkpoint
    print('Loading best model..')
    checkpoint = torch.load(os.path.join(path, "best_model.pt"))
    model.bert.load_state_dict(checkpoint['bert_state_dict'])
    for index, optimizer in enumerate(optimizers):
        optimizer.load_state_dict(checkpoint['optimizer_state_dicts'][index])
    print('Best model loaded')

    # return the model, optimizers and results
    return model, optimizers, gathered_results


def handle_matched(args, device, path):
    """
    Function handling the matched setting.
    Inputs:
        args - Namespace object from the argument parser
        device - PyTorch device to use
        path - Path for storing the results
    """

    # load the model
    print('Loading model..')
    model, tokenizer, optimizers = initialize_model(args, device)
    print('Model loaded')

    # load the datasets
    print('Loading datasets..')
    train_set, dev_set, test_set = LoadCircaMatched(args, tokenizer)
    train_set = {'Circa': train_set}
    dev_set = {'Circa': dev_set}
    test_set = {'Circa': test_set}
    for task in args.aux_tasks:
        if task == 'SST2':
            train_aux_set, dev_aux_set, test_aux_set = LoadSST2(args, tokenizer)
        elif task == 'MNLI':
            train_aux_set, dev_aux_set, test_aux_set = LoadMNLI(args, tokenizer)
        elif task == 'BOOLQ':
            train_aux_set, dev_aux_set, test_aux_set = LoadBoolQ(args, tokenizer)
        elif task == 'IQAP':
            train_aux_set, dev_aux_set, test_aux_set = LoadIQAP(args, tokenizer)
        # TODO: add all other datasets
        train_set[task] = train_aux_set
        dev_set[task] = dev_aux_set
        test_set[task] = test_aux_set

    # combine the dataloaders into a multi task datalaoder
    train_set = MultiTaskDataloader(dataloaders=train_set)
    dev_set = MultiTaskDataloader(dataloaders=dev_set)
    test_set = MultiTaskDataloader(dataloaders=test_set)
    print('Datasets loaded')

    # check if a checkpoint is provided
    if args.checkpoint_path is not None:
        # load the model from the given checkpoint
        print('Loading model from checkpoint..')
        checkpoint = torch.load(args.checkpoint_path)
        model.bert.load_state_dict(checkpoint['bert_state_dict'])
        for index, optimizer in enumerate(optimizers):
            optimizer.load_state_dict(checkpoint['optimizer_state_dicts'][index])
        print('Model loaded')
    else:
        # train the model
        model, optimizers, gathered_results = train_model(
            args = args,
            model = model,
            optimizers=optimizers,
            train_set = train_set,
            dev_set = dev_set,
            test_set = test_set,
            device = device,
            path = path
        )

    # test the model
    print('Starting testing..')
    with torch.no_grad():
        test_results = perform_epoch(args, model, optimizers, test_set, device, train=False)
    print('Test results:')
    print(test_results)
    print('Testing finished')

    # save the testing measures
    if args.checkpoint_path is None:
        gathered_results['testing'] = test_results

        # save the results as a json file
        print('Saving results..')
        with open(os.path.join(path, 'results.txt'), 'w') as outfile:
            json.dump(gathered_results, outfile)
        print('Results saved')


def handle_unmatched(args, device, path):
    """
    Function handling the unmatched setting.
    Inputs:
        args - Namespace object from the argument parser
        device - PyTorch device to use
        path - Path for storing the results
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

    # train and test the model for each held-out scenario
    for index, test_scenario in enumerate(scenarios):
        # pick a random development scenario
        left_scenarios = scenarios[:]
        left_scenarios.remove(test_scenario)
        dev_scenario = random.choice(left_scenarios)

        # load the model
        print('Loading model..')
        model, tokenizer, optimizers = initialize_model(args, device)
        print('Model loaded')

        # load the datasets
        print('Loading datasets..')
        train_set, dev_set, test_set = LoadCircaUnmatched(args, tokenizer, test_scenario, dev_scenario)
        train_set = {'Circa': train_set}
        dev_set = {'Circa': dev_set}
        test_set = {'Circa': test_set}
        for task in args.aux_tasks:
            if task == 'SST2':
                train_aux_set, dev_aux_set, test_aux_set = LoadSST2(args, tokenizer)
            elif task == 'MNLI':
                train_aux_set, dev_aux_set, test_aux_set = LoadMNLI(args, tokenizer)
            elif task == 'BOOLQ':
                train_aux_set, dev_aux_set, test_aux_set = LoadBoolQ(args, tokenizer)
            elif task == 'IQAP':
                train_aux_set, dev_aux_set, test_aux_set = LoadIQAP(args, tokenizer)
            # TODO: add all other datasets
            train_set[task] = train_aux_set
            dev_set[task] = dev_aux_set
            test_set[task] = test_aux_set

        # combine the dataloaders into a multi task datalaoder
        train_set = MultiTaskDataloader(dataloaders=train_set)
        dev_set = MultiTaskDataloader(dataloaders=dev_set)
        test_set = MultiTaskDataloader(dataloaders=test_set)
        print('Datasets loaded')

        # train the model
        model, optimizers, gathered_results = train_model(
            args = args,
            model = model,
            optimizers=optimizers,
            train_set = train_set,
            dev_set = dev_set,
            test_set = test_set,
            device = device,
            path = path
        )

        # test the model
        print('Starting testing..')
        with torch.no_grad():
            test_results = perform_epoch(args, model, optimizers, test_set, device, train=False)
        print('Test results:')
        print(test_results)
        print('Testing finished')

        # save the results for the current scenario
        gathered_results['testing'] = test_results
        test_results['scenario' + str(index + 1)] = gathered_results

    # save the results as a json file
    print('Saving results..')
    with open(os.path.join(path, 'results.txt'), 'w') as outfile:
        json.dump(test_results, outfile)
    print('Results saved')


def main(args):
    """
    Function for handling the arguments and starting the experiment.
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
    print('Setting: {}'.format(args.setting))
    print('Auxilary tasks: {}'.format(args.aux_tasks))
    print('Auxilary task probing: {}'.format(args.aux_probing))
    print('PyTorch device: {}'.format(device))
    print('Max epochs: {}'.format(args.max_epochs))
    print('Patience: {}'.format(args.patience))
    print('Learning rate: {}'.format(args.lr))
    print('Batch size: {}'.format(args.batch_size))
    print('Results directory: {}'.format(args.results_dir))
    print('Progress bar: {}'.format(args.progress_bar))
    print('-----------------------------')

    # generate the path to use for the results
    path = create_path(args)
    if not os.path.exists(path):
        os.makedirs(path)

    # check which setting is selected
    if args.setting == 'matched':
        handle_matched(args, device, path)
    else:
        handle_unmatched(args, device, path)


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
    parser.add_argument('--max_epochs', default=10, type=int,
                        help='Maximum number of epochs to train for. Default is 10')
    parser.add_argument('--patience', default=3, type=int,
                        help='Stops training after patience number of epochs without improvement in dev accuracy. Default is 3')

    # optimizer hyperparameters
    parser.add_argument('--lr', default=3e-5, type=float,
                        help='Learning rate to use. Default is 3e-5')
    parser.add_argument('--batch_size', default=8, type=int,
                        help='Minibatch size. Default is 8')

    # mtl hyperparameters
    parser.add_argument('--aux_tasks', default=[], type=str, nargs='*',
                        help='Which auxilary tasks to train on. Default is [] (STL)',
                        choices=['IQAP', 'SST2', 'MNLI', 'BOOLQ'])
    parser.add_argument('--aux_probing', action='store_true',
                        help=('Does not train BERT on the auxilary tasks, but only the classification layer.'))

    # loading hyperparameters
    parser.add_argument('--checkpoint_path', default=None, type=str,
                        help='Path to where the model checkpoint is located. Default is None (no checkpoint used)')

    # other hyperparameters
    parser.add_argument('--seed', default=1234, type=int,
                        help='Seed to use for reproducing results. Default is 1234')
    parser.add_argument('--results_dir', default='./mtl_results', type=str,
                        help='Directory where the training results should be created. Default is ./mtl_results')
    parser.add_argument('--progress_bar', action='store_true',
                        help=('Use a progress bar indicator for interactive experimentation. '
                              'Not to be used in conjuction with SLURM jobs'))

    # parse the arguments
    args = parser.parse_args()

    # train the model
    main(args)
