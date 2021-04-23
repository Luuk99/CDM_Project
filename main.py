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
from data.LoadCircaData import load_circa_matched, load_circa_unmatched
from data.LoadSST2Data import load_sst2
from utils import create_dataloader, compute_accuracy, create_path, initialize_model

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

    # pass the batch through the model
    outputs = model(input_ids, attention_mask=attention_mask, labels=batch_labels, task_idx=task_idx)
    loss = outputs.loss

    if train:
        # optimize the loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    # return the loss, label and prediction
    return loss, outputs.logits, batch_labels


def perform_epoch(args, model, main_optimizer, aux_optimizer, dataset, aux_dataset, device, train=True):
    """
    Function that performs an epoch for the given model.
    Inputs:
        args - Namespace object from the argument parser
        model - BERT model instance
        main_optimizer - AdamW optimizer instance for the main task
        aux_optimizer - AdamW optimizer instance for auxilary task
        dataset - Dataset to use
        aux_dataset - Dataset for the second task to use
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
    main_predictions = []
    main_labels = []
    main_losses = []
    aux_predictions = []
    aux_labels = []
    aux_losses = []

    # create a list for the auxilary dataset
    if args.aux_task is not None:
        aux_dataset = [batch for _, batch in enumerate(aux_dataset)]

    # loop over the batches
    if (args.progress_bar):
        dataset = tqdm(dataset)
    for step, batch in enumerate(dataset):
        # perform a step for the main task
        main_step_loss, main_step_predictions, main_step_labels = perform_step(model, main_optimizer, batch, device, 0, train)
        main_predictions.append(main_step_predictions)
        main_labels.append(main_step_labels)
        main_losses.append(main_step_loss)

        # perform a step for the auxilary task if an auxilary task is specified
        if (args.aux_task is not None and step < len(aux_dataset)):
            aux_step_loss, aux_step_predictions, aux_step_labels = perform_step(model, aux_optimizer, aux_dataset[step], device, 1, train)
            aux_predictions.append(aux_step_predictions.squeeze())
            aux_labels.append(aux_step_labels)
            aux_losses.append(aux_step_loss)

    # calculate the loss and accuracy for the main task
    mean_main_loss = torch.mean(torch.stack(main_losses, dim=0), dim=0)
    mean_main_loss = round(mean_main_loss.item(), 4)
    main_accuracy = compute_accuracy(main_predictions, main_labels)

    # calculate the loss and accuracy for the auxilary task
    if args.aux_task is not None:
        mean_aux_loss = torch.mean(torch.stack(aux_losses, dim=0), dim=0)
        mean_aux_loss = round(mean_aux_loss.item(), 4)
        aux_accuracy = compute_accuracy(aux_predictions, aux_labels)
    else:
        mean_aux_loss, aux_accuracy = (None, None)

    # record the end time
    end_time = timer()

    # calculate the elapsed time
    elapsed_time = str(datetime.timedelta(seconds=(end_time - start_time)))

    # return the elapsed time, loss and accuracy
    return elapsed_time, mean_main_loss, main_accuracy, mean_aux_loss, aux_accuracy


def train_model(args, model, main_optimizer, aux_optimizer, train_set, dev_set, test_set, train_aux_set, dev_aux_set, test_aux_set, device, path):
    """
    Function for training the model.
    Inputs:
        args - Namespace object from the argument parser
        model - BERT model instance
        main_optimizer - AdamW optimizer instance for the main task
        aux_optimizer - AdamW optimizer instance for auxilary task
        train_set - Circa training set
        dev_set - Circa development set
        test_set - Circa test set
        train_aux_set - Auxilary task training set
        dev_aux_set - Auxilary task development set
        test_aux_set - Auxilary task test set
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
        dev_time, main_dev_loss, main_dev_acc, aux_dev_loss, aux_dev_acc = perform_epoch(args, model, main_optimizer, aux_optimizer, dev_set, dev_aux_set, device, train=False)
    print('Time: dev_time {} \nMain performance: dev_loss {} & dev_acc {} \nAuxilary performance: dev_loss {} & dev_acc {}'.format(
            dev_time,
            main_dev_loss,
            main_dev_acc,
            aux_dev_loss,
            aux_dev_acc,
        )
    )

    # save the pre-training evaluation measures
    gathered_results['epoch0'] = {
        'time': {'dev_time': dev_time},
        'main_performance' : {
            'dev_loss': main_dev_loss,
            'dev_acc': main_dev_acc,
        },
        'aux_performance' : {
            'dev_loss': aux_dev_loss,
            'dev_acc': aux_dev_acc,
        }
    }

    # train the model
    best_dev_acc = 0
    epochs_no_improvement = 0
    for epoch in range(1, args.max_epochs + 1):
        print('Epoch {}:'.format(epoch))

        # perform a training epoch
        train_time, main_train_loss, main_train_acc, aux_train_loss, aux_train_acc = perform_epoch(args, model, main_optimizer, aux_optimizer, train_set, train_aux_set, device, train=True)

        # perform a development epoch
        with torch.no_grad():
            dev_time, main_dev_loss, main_dev_acc, aux_dev_loss, aux_dev_acc = perform_epoch(args, model, main_optimizer, aux_optimizer, dev_set, dev_aux_set, device, train=False)

        # print the epoch measures
        print('Time: train_time {} & dev_time {} \nMain performance: train_loss {} & train_acc {} & dev_loss {} & dev_acc {} \nAuxilary performance: train_loss {} & train_acc {} & dev_loss {} & dev_acc {}'.format(
                train_time,
                dev_time,
                main_train_loss,
                main_train_acc,
                main_dev_loss,
                main_dev_acc,
                aux_train_loss,
                aux_train_acc,
                aux_dev_loss,
                aux_dev_acc,
            )
        )

        # save the epoch measures
        gathered_results['epoch' + str(epoch)] = {
            'time': {'train_time': train_time, 'dev_time': dev_time},
            'main_performance' : {
                'train_loss': main_train_loss,
                'train_acc': main_train_acc,
                'dev_loss': main_dev_loss,
                'dev_acc': main_dev_acc,
            },
            'aux_performance' : {
                'train_loss': aux_train_loss,
                'train_acc': aux_train_acc,
                'dev_loss': aux_dev_loss,
                'dev_acc': aux_dev_acc
            }
        }

        # check whether to save the model or not
        if (round(main_dev_acc, 2) > best_dev_acc):
            epochs_no_improvement = 0
            best_dev_acc = round(main_dev_acc, 2)
            print('Saving new best model..')
            if args.aux_task is not None:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'main_optimizer_state_dict': main_optimizer.state_dict(),
                    'aux_optimizer_state_dict': aux_optimizer.state_dict(),
                }, os.path.join(path, "best_model.pt"))
            else:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'main_optimizer_state_dict': main_optimizer.state_dict(),
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
    model.load_state_dict(checkpoint['model_state_dict'])
    main_optimizer.load_state_dict(checkpoint['main_optimizer_state_dict'])
    if args.aux_task is not None:
        aux_optimizer.load_state_dict(checkpoint['aux_optimizer_state_dict'])
    print('Best model loaded')

    # return the model, optimizers and results
    return model, main_optimizer, aux_optimizer, gathered_results


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
    model, tokenizer, main_optimizer, aux_optimizer = initialize_model(args, device)
    print('Model loaded')

    # load the dataset
    print('Loading dataset..')
    train_set, dev_set, test_set = load_circa_matched(args, tokenizer)
    if args.aux_task == 'SST':
        train_aux_set, dev_aux_set, test_aux_set = load_sst2(args, tokenizer)
    # TODO: add all other datasets
    else:
        train_aux_set, dev_aux_set, test_aux_set = (None, None, None)
    print('Dataset loaded')

    # check if a checkpoint is provided
    if args.checkpoint_path is not None:
        # load the model from the given checkpoint
        print('Loading model from checkpoint..')
        checkpoint = torch.load(args.checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        main_optimizer.load_state_dict(checkpoint['main_optimizer_state_dict'])
        aux_optimizer.load_state_dict(checkpoint['aux_optimizer_state_dict'])
        print('Model loaded')
    else:
        # train the model
        model, main_optimizer, aux_optimizer, gathered_results = train_model(
            args = args,
            model = model,
            main_optimizer = main_optimizer,
            aux_optimizer = aux_optimizer,
            train_set = train_set,
            dev_set = dev_set,
            test_set = test_set,
            train_aux_set = train_aux_set,
            dev_aux_set = dev_aux_set,
            test_aux_set = test_aux_set,
            device = device,
            path = path
        )

    # test the model
    print('Starting testing..')
    with torch.no_grad():
        test_time, main_test_loss, main_test_acc, aux_test_loss, aux_test_acc = perform_epoch(args, model, main_optimizer, aux_optimizer, test_set, train_aux_set, device, train=False)
    print('Time: test_time {} \nMain performance: test_loss {} & test_acc {} \nAuxilary performance: test_loss {} & test_acc {}'.format(
            test_time,
            main_test_loss,
            main_test_acc,
            aux_test_loss,
            aux_test_acc,
        )
    )
    print('Testing finished')

    # save the testing measures
    if args.checkpoint_path is None:
        gathered_results['testing'] = {
            'time': {'test_time': test_time},
            'main_performance' : {
                'dev_loss': main_test_loss,
                'dev_acc': main_test_acc,
            },
            'aux_performance' : {
                'dev_loss': aux_test_loss,
                'dev_acc': aux_test_acc,
            }
        }

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
        model, tokenizer, main_optimizer, aux_optimizer = initialize_model(args, device)
        print('Model loaded')

        # load the dataset
        print('Loading dataset..')
        train_set, dev_set, test_set = load_circa_unmatched(args, tokenizer, test_scenario, dev_scenario)
        if args.aux_task == 'SST':
            train_aux_set, dev_aux_set, test_aux_set = load_sst2(args, tokenizer)
        # TODO: add all other datasets
        else:
            train_aux_set, dev_aux_set, test_aux_set = (None, None, None)
        print('Dataset loaded')

        # train the model
        model, main_optimizer, aux_optimizer, gathered_results = train_model(
            args = args,
            model = model,
            main_optimizer = main_optimizer,
            aux_optimizer = aux_optimizer,
            train_set = train_set,
            dev_set = dev_set,
            test_set = test_set,
            train_aux_set = train_aux_set,
            dev_aux_set = dev_aux_set,
            test_aux_set = test_aux_set,
            device = device,
            path = path
        )

        # test the model
        print('Starting testing..')
        with torch.no_grad():
            test_time, main_test_loss, main_test_acc, aux_test_loss, aux_test_acc = perform_epoch(args, model, main_optimizer, aux_optimizer, test_set, train_aux_set, device, train=False)
        print('Time: test_time {} \nMain performance: test_loss {} & test_acc {} \nAuxilary performance: test_loss {} & test_acc {}'.format(
                test_time,
                main_test_loss,
                main_test_acc,
                aux_test_loss,
                aux_test_acc,
            )
        )
        print('Testing finished')

        # save the testing measures
        gathered_results['testing'] = {
            'time': {'test_time': test_time},
            'main_performance' : {
                'dev_loss': main_test_loss,
                'dev_acc': main_test_acc,
            },
            'aux_performance' : {
                'dev_loss': aux_test_loss,
                'dev_acc': aux_test_acc,
            }
        }

        # save the results for the current scenario
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
    print('Auxilary task: {}'.format(args.aux_task))
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
    parser.add_argument('--aux_task', default=None, type=str,
                        help='Which auxilary task to train on. Default is None (STL)',
                        choices=['SST2'])
    parser.add_argument('--aux_probing', action='store_true',
                        help=('Does not train BERT on the auxilary task, but only the classification layer.'))

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
