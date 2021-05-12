# imports
import os
import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# huggingface imports
import transformers
from transformers import BertTokenizer
from transformers import AdamW
from transformers.data.data_collator import DataCollatorWithPadding

# own imports
from multi_task.bert_mtl import MLTBertForSequenceClassification


def create_path(args):
    """
    Function that creates a path for the results based on the model arguments.
    Inputs:
        args - Namespace object from the argument parser
    Outputs:
        path - Path where to store the results
    """

    # create the path
    if not args.aux_tasks:
        aux_task = 'no_aux'
    else:
        aux_task = ''
        for task in args.aux_tasks:
            aux_task += (task + '_')
        aux_task += 'probing' if args.aux_probing else 'trained'
    path = os.path.join(
        args.results_dir,
        args.model_version,
        args.labels,
        args.setting,
        aux_task
    )

    # return the path
    return path


def initialize_model(args, device):
    """
    Function that initializes the model, tokenizer and optimizers.
    Inputs:
        args - Namespace object from the argument parser
        device - PyTorch device to use
    Outputs:
        model - MultiTask BERT model instance
        tokenizer - BERT tokenizer instance
        optimizers - List of optimizers
    """

    # check if the learning rates list is as long as the number of tasks
    if (len(args.lrs) != (len(args.aux_tasks) + 1)):
        raise ValueError('Length of learning rates list must match the number of tasks (auxiliary + main)')

    # dictionary for the number of classes per task
    # TODO: add all tasks here
    task_label_dict = {
        'SST2': 1,
        'MNLI': 3,
        'BOOLQ': 2,
        'IQAP': 4,
    }

    # check how many labels to use
    if args.labels == 'strict':
        num_labels = 6
    else:
        num_labels = 4

    # load the model
    model = MLTBertForSequenceClassification.from_pretrained('bert-base-uncased',
        num_labels=num_labels
    ).to(device)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # add the auxilary tasks
    aux_task_labels = [task_label_dict[task] for task in args.aux_tasks]
    model.add_aux_classifiers(aux_task_labels)

    # create the optimizers
    optimizers = []
    #no_decay = ['bias', 'gamma', 'beta', 'LayerNorm.bias', 'LayerNorm.weight']
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.bert.named_parameters() if not any(nd in n for nd in no_decay)]
        + [p for n, p in model.classifiers[0].named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in model.bert.named_parameters() if any(nd in n for nd in no_decay)]
        + [p for n, p in model.classifiers[0].named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    # add the main task optimizer
    optimizers.append(AdamW(optimizer_grouped_parameters, lr=args.lrs[0]))
    # add the auxiliary task optimizers
    for index, task in enumerate(args.aux_tasks):
        if args.aux_probing:
            optimizer_grouped_parameters = [
                {'params': [p for n, p in model.classifiers[index + 1].named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
                {'params': [p for n, p in model.classifiers[index + 1].named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]
        else:
            optimizer_grouped_parameters = [
                {'params': [p for n, p in model.bert.named_parameters() if not any(nd in n for nd in no_decay)]
                + [p for n, p in model.classifiers[index + 1].named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
                {'params': [p for n, p in model.bert.named_parameters() if any(nd in n for nd in no_decay)]
                + [p for n, p in model.classifiers[index + 1].named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]
        optimizers.append(AdamW(optimizer_grouped_parameters, lr=args.lrs[index + 1]))

    # return the model, tokenizer and optimizers
    return model, tokenizer, optimizers


def create_dataloader(args, dataset, tokenizer):
    """
    Function to create a PyTorch Dataloader from a given dataset.
    Inputs:
        args - Namespace object from the argument parser
        dataset - Dataset to convert to Dataloader
        tokenizer - BERT tokenizer instance
    Outputs:
        dataset - DataLoader object of the dataset
    """

    # create a data collator function
    data_collator = DataCollatorWithPadding(tokenizer)

    # create the dataloader
    dataset = DataLoader(
        dataset,
        batch_size=args.batch_size,
        collate_fn=data_collator,
        drop_last=False,
        shuffle=True,
    )

    # return the dataset
    return dataset


def compute_accuracy(preds, labels):
    """
    Function that calculates the accuracy
    Inputs:
        preds - List of batched predictions from the model
        labels - List of batched real labels
    Outputs:
        acc - Accuracy of the predictions and real labels
    """

    # concatenate the predictions and labels
    preds = torch.cat(preds, dim=0).squeeze()
    labels = torch.cat(labels, dim=0).squeeze()

    # check if regression or classification
    if len(preds.shape) > 1:
        preds = torch.nn.functional.softmax(preds, dim=-1)
        preds = torch.argmax(preds, dim=-1)
    else:
        preds = torch.round(preds)
        labels = torch.round(labels)

    # calculate the accuracy
    acc = accuracy_score(labels.cpu().detach(), preds.cpu().detach())

    # round to 4 decimals
    acc = round(acc, 4)

    # return the accuracy
    return acc


def handle_epoch_metrics(step_metrics):
    """
    Function that handles the metrics per epoch.
    Inputs:
        step_metrics - Dictionary containing the results of the steps of an epoch
    Outputs:
        epoch_merics - Dictionary containing the averaged results of an epoch
    """

    # create a new dictionary with epoch results
    epoch_metrics = {}

    # loop over the tasks in the step metrics dictionary
    for task in step_metrics:
        # compute the loss
        task_loss = torch.mean(torch.stack(step_metrics[task]['losses'], dim=0), dim=0)
        task_loss = round(task_loss.item(), 4)

        # compute the accuracy
        task_accuracy = compute_accuracy(step_metrics[task]['predictions'], step_metrics[task]['labels'])

        # add to the epoch dictionary
        epoch_metrics[task] = {'loss': task_loss, 'accuracy': task_accuracy}

    # return the epoch dictionary
    return epoch_metrics

def str2bool(v):
    """
    Useful for bool argparsing, adopted from: https://stackoverflow.com/a/43357954/4022008
    """
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')