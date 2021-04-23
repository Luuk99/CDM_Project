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
    """

    # create the path
    probing = '_probing' if args.aux_probing else '_trained'
    aux_task = 'no_aux' if (args.aux_task is None) else (args.aux_task + probing)
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
    """

    # check how many labels to use
    if args.labels == 'strict':
        num_labels = 6
    else:
        num_labels = 4

    # load the model
    model = MLTBertForSequenceClassification.from_pretrained('bert-base-uncased',
        num_labels=num_labels
    ).to(device)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', sep_token='[SEP]', unk_token='[UNK]')

    # add the auxilary tasks
    if args.aux_task == 'SST2':
        model.add_aux_classifiers(1)
    # TODO: add all tasks here

    # create the optimizers
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    main_optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr)
    if args.aux_probing:
        optimizer_grouped_parameters = model.classifiers[1].parameters()
    if args.aux_task is not None:
        aux_optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr)
    else:
        aux_optimizer = None

    # return the model, tokenizer and optimizers
    return model, tokenizer, main_optimizer, aux_optimizer


def create_dataloader(args, dataset, tokenizer):
    """
    Function to create a PyTorch Dataloader from a given dataset.
    Inputs:
        args - Namespace object from the argument parser
        dataset - Dataset to convert to Dataloader
        tokenizer - BERT tokenizer instance
    """

    # create a sampler
    sampler = RandomSampler(dataset)

    # create a data collator function
    data_collator = DataCollatorWithPadding(tokenizer)

    # create the dataloader
    dataset = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        collate_fn=data_collator,
        drop_last=False,
    )

    # return the dataset
    return dataset


def compute_accuracy(preds, labels):
    """
    Function that calculates the accuracy
    Inputs:
        preds - List of batched predictions from the model
        labels - List of batched real labels
    """

    # concatenate the predictions and labels
    preds = torch.cat(preds, dim=0).squeeze()
    labels = torch.cat(labels, dim=0).squeeze()

    # check if regression or classification
    if len(preds.shape) > 1:
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
