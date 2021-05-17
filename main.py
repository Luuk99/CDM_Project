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
from data.load_circa_data import LoadCirca
from data.load_sst2_data import LoadSST2
from data.load_mnli_data import LoadMNLI
from data.load_boolq_data import LoadBoolQ
from data.load_iqap_data import LoadIQAP
from data.multitask_dataloader import MultiTaskDataloader
from utils import create_dataloader, handle_epoch_metrics, create_path, initialize_model_optimizers, initialize_tokenizer, str2bool

# set Huggingface logging to error only
import transformers
transformers.logging.set_verbosity_error()


def perform_step(model, optimizer, batch, device, task_idx, train=True, aux_probing=False):
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

    # check whether we are probing
    if aux_probing and (task_idx != 0):
        # freeze the BERT parameters
        for parameter in model.bert.parameters():
            parameter.requires_grad = False
    else:
        # unfreeze the BERT parameters
        for parameter in model.bert.parameters():
            parameter.requires_grad = True

    # pass the batch through the model
    outputs = model(input_ids, attention_mask=attention_mask, labels=batch_labels, token_type_ids=token_type_ids, task_idx=task_idx)
    loss = outputs.loss

    if train:
        # backward using the loss
        loss.backward()

        # set a step with the optimizer
        optimizer.step()
        optimizer.zero_grad()

    # return the loss, label and prediction
    return loss, outputs.logits, batch_labels


def perform_epoch(args, model, optimizers, dataset, device, train=True, advanced_metrics=False):
    """
    Function that performs an epoch for the given model.
    Inputs:
        args - Namespace object from the argument parser
        model - BERT model instance
        optimizers - List of optimizers to use
        dataset - Dataset to use
        device - PyTorch device to use
        train - Whether to train or test the model
        advanced_metrics - Whether to calculate confusion matrices and f1 scores
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
        step_loss, step_predictions, step_labels = perform_step(model, optimizers[task_idx], batch, device, task_idx, train, args.aux_probing)

        # add the results to the dictionary
        if task_name in result_dict:
            result_dict[task_name]['predictions'].append(step_predictions)
            result_dict[task_name]['labels'].append(step_labels)
            result_dict[task_name]['losses'].append(step_loss)
        else:
            result_dict[task_name] = {
                'predictions': [step_predictions],
                'labels': [step_labels],
                'losses': [step_loss]
            }

    # calculate the loss and accuracy for the different tasks
    epoch_results = handle_epoch_metrics(result_dict, advanced_metrics)

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
        if (round(dev_results['Circa']['accuracy'], 3) > best_dev_acc):
            epochs_no_improvement = 0
            best_dev_acc = round(dev_results['Circa']['accuracy'], 3)
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

    # load the tokenizer
    tokenizer = initialize_tokenizer()

    # load the datasets
    print('Loading datasets..')
    topicLabelCount = 0
    train_set, dev_set, test_set, label_dict = LoadCirca(args, tokenizer)
    for task in args.aux_tasks:
        if task == 'SST2':
            train_aux_set, dev_aux_set, test_aux_set = LoadSST2(args, tokenizer)
        elif task == 'MNLI':
            train_aux_set, dev_aux_set, test_aux_set = LoadMNLI(args, tokenizer)
        elif task == 'BOOLQ':
            train_aux_set, dev_aux_set, test_aux_set = LoadBoolQ(args, tokenizer)
        elif task == 'IQAP':
            train_aux_set, dev_aux_set, test_aux_set = LoadIQAP(args, tokenizer)
        elif task == 'TOPICS':
            topicLabelCount = len(label_dict['TOPICS'])
            continue # TOPICS aux task will be loaded automatically
        # TODO: add all other datasets
        train_set[task] = train_aux_set
        dev_set[task] = dev_aux_set
        test_set[task] = test_aux_set

    # combine the dataloaders into a multi task datalaoder
    train_set = MultiTaskDataloader(dataloaders=train_set)
    dev_set = MultiTaskDataloader(dataloaders=dev_set)
    test_set = MultiTaskDataloader(dataloaders=test_set)
    print('Datasets loaded')

    # load the model
    print('Loading model..')
    model, optimizers = initialize_model_optimizers(args, device, topicLabelCount)
    print('Model loaded')

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
        test_results = perform_epoch(args, model, optimizers, test_set, device, train=False, advanced_metrics=args.advanced_metrics)
        print(label_dict)
    print('Test results:')
    print(test_results)
    print('Testing finished')

    # save the testing measures
    if args.checkpoint_path is None:
        gathered_results['testing'] = test_results
        gathered_results['label_dict'] = label_dict['Circa']
        
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

    # select the test_scenario
    test_scenario = scenarios[args.test_scenario]

    # load the tokenizer
    tokenizer = initialize_tokenizer()

    # load the datasets
    print('Loading datasets..')
    topicLabelCount = 0
    train_set, dev_set, test_set, label_dict = LoadCirca(args, tokenizer, test_scenario)
    for task in args.aux_tasks:
        if task == 'SST2':
            train_aux_set, dev_aux_set, test_aux_set = LoadSST2(args, tokenizer)
        elif task == 'MNLI':
            train_aux_set, dev_aux_set, test_aux_set = LoadMNLI(args, tokenizer)
        elif task == 'BOOLQ':
            train_aux_set, dev_aux_set, test_aux_set = LoadBoolQ(args, tokenizer)
        elif task == 'IQAP':
            train_aux_set, dev_aux_set, test_aux_set = LoadIQAP(args, tokenizer)
        elif task == 'TOPICS':
            topicLabelCount = len(label_dict['TOPICS'])
            continue # TOPICS aux task will be loaded automatically
            
        # TODO: add all other datasets
        train_set[task] = train_aux_set
        dev_set[task] = dev_aux_set
        test_set[task] = test_aux_set

    # combine the dataloaders into a multi task datalaoder
    train_set = MultiTaskDataloader(dataloaders=train_set)
    dev_set = MultiTaskDataloader(dataloaders=dev_set)
    test_set = MultiTaskDataloader(dataloaders=test_set)
    print('Datasets loaded')

    # load the model
    print('Loading model..')
    model, optimizers = initialize_model_optimizers(args, device, topicLabelCount)
    print('Model loaded')

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
        test_results = perform_epoch(args, model, optimizers, test_set, device, train=False, advanced_metrics=args.advanced_metrics)
        print(label_dict)
    print('Test results:')
    print(test_results)
    print('Testing finished')

    # save the testing measures
    print('Saving results..')
    with open(os.path.join(path, 'results.txt'), 'w') as outfile:
        json.dump(gathered_results, outfile)
    print('Results saved')

    # save the results as a json file
    gathered_results['testing'] = test_results
    gathered_results['label_dict'] = label_dict['Circa']
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
    print('Test scenario: {}'.format(args.test_scenario))
    print('Auxilary tasks: {}'.format(args.aux_tasks))
    print('Auxilary task probing: {}'.format(args.aux_probing))
    print('PyTorch device: {}'.format(device))
    print('Max epochs: {}'.format(args.max_epochs))
    print('Patience: {}'.format(args.patience))
    print('Learning rates: {}'.format(args.lrs))
    print('Batch size: {}'.format(args.batch_size))
    print('Results directory: {}'.format(args.results_dir))
    print('Progress bar: {}'.format(args.progress_bar))
    print('Advanced metrics: {}'.format(args.advanced_metrics))
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
    parser.add_argument('--test_scenario', default=None, type=int,
                        help='What test scenario to use. Only use in combination with setting unmatched. Default is None',
                        choices=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

    # annotation options (note: these do NOT initiate an auxiliary task, but adding 'TOPICS' under --aux_tasks DOES initiate (default values) of these vars)
    parser.add_argument('--impwords', nargs='?', type=str2bool, const=True, default=False,
                        help='If mentioned, Circa dataset will be annotated with most important word in answers.')
    parser.add_argument('--topics', nargs='?', type=str2bool, const=True, default=False,
                        help='If mentioned, Circa dataset will be annotated with a WordNet topic for every answer')
    parser.add_argument('--npimpwords', nargs='?', type=str2bool, const=False, default=True,
                        help='If mentioned, important words annotations will NOT be pre-loaded, but re-generated')
    parser.add_argument('--nptopics', nargs='?', type=str2bool, const=False, default=True,
                        help='If mentioned, topic annotations will NOT be pre-loaded, but re-generated')
    parser.add_argument('--tfidf', nargs='?', type=str2bool, const=True, default=False,
                        help='If mentioned, most important words will be determined by TF-IDF values as opposed to extracting the last noun')
    parser.add_argument('--hybrid', nargs='?', type=str2bool, const=True, default=False,
                        help='If mentioned, most important words will be determined by TF-IDF values ONLY if there is no last noun')
    parser.add_argument('--traversetopics', nargs='?', type=str2bool, const=True, default=False,
                        help='If mentioned, topic annotations will be generated using all-hypernym traversal')
    parser.add_argument('--topic_depth', default=None, type=int,
                        help='Top-down tree depth for naive case without tree traversing')
    parser.add_argument('--label_density', default=None, type=int,
                        help='Controls the level of allowed topic class labels')
    parser.add_argument('--impwordsfile', default=None, type=str,
                        help='Plain-text important words annotation file per indirect answer. Default is fixed in annotate_circa_data.py')
    parser.add_argument('--topicsfile', default=None, type=str,
                        help='Plain-text topic annotation file per indirect answer. Default is fixed in annotate_circa_data.py')
    parser.add_argument('--topiclabelsfile', default=None, type=str,
                        help='Pickled topic label annotation file per indirect answer. Default is fixed in annotate_circa_data.py')

    # training hyperparameters
    parser.add_argument('--max_epochs', default=5, type=int,
                        help='Maximum number of epochs to train for. Default is 5')
    parser.add_argument('--patience', default=3, type=int,
                        help='Stops training after patience number of epochs without improvement in dev accuracy. Default is 3')

    # optimizer hyperparameters
    parser.add_argument('--lrs', default=[3e-5], type=float, nargs='*',
                        help='Learning rates to use per task. Default is [3e-5] (STL)')
    parser.add_argument('--batch_size', default=8, type=int,
                        help='Minibatch size. Default is 8')

    # mtl hyperparameters
    parser.add_argument('--aux_tasks', default=[], type=str, nargs='*',
                        help='Which auxiliary tasks to train on. Default is [] (STL)',
                        choices=['IQAP', 'SST2', 'MNLI', 'BOOLQ', 'TOPICS'])
    parser.add_argument('--aux_probing', action='store_true',
                        help=('Does not train BERT on the auxiliary tasks, but only the classification layer.'))

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
    parser.add_argument('--advanced_metrics', action='store_true',
                        help='Generate confusion matrices and f1 scores.')

    # parse the arguments
    args = parser.parse_args()

    # invoke implied arguments
    if 'TOPICS' in args.aux_tasks and (not args.impwords or not args.topics):
        argsModified = vars(args)
        argsModified['impwords'] = True
        argsModified['topics']   = True

    # train the model
    main(args)
